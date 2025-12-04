package projectstalker.physics.jni;

import lombok.extern.slf4j.Slf4j;
import projectstalker.domain.river.RiverGeometry;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

/**
 * Actúa como un puente (wrapper) Stateful hacia la librería nativa JNI para resolver
 * la ecuación de Manning en la GPU.
 * <p>
 * REFACTORIZADO PARA DMA (Direct Memory Access):
 * Mantiene buffers de memoria "Pinned" (DirectByteBuffer) persistentes para reutilizarlos
 * en cada batch, eliminando la fragmentación del Heap y las copias de JNI.
 */
@Slf4j
public final class ManningGpuSolver implements AutoCloseable {

    private final INativeManningSolver nativeSolver;
    private long sessionHandle = 0; // Puntero a la sesión C++ (0 = inválido)
    private final int cellCount;
    private final RiverGeometry geometry;

    // --- DMA Buffers (Reutilizables) ---
    // Se asignan una vez y crecen solo si es necesario.
    // Al ser DirectBuffers, residen fuera del Garbage Collector y tienen dirección física fija.
    private FloatBuffer inputBuffer = null;
    private FloatBuffer outputBuffer = null;

    public ManningGpuSolver(RiverGeometry geometry) {
        this(NativeManningGpuSingleton.getInstance(), geometry);
    }

    public ManningGpuSolver(INativeManningSolver nativeSolver, RiverGeometry geometry) {
        this.nativeSolver = nativeSolver;
        this.geometry = geometry;
        this.cellCount = geometry.getCellCount();
    }

    /**
     * Inicializa la sesión GPU (Lazy Init).
     * Mantiene la lógica original de crear buffers temporales solo para la carga inicial.
     */
    private void initializeSession(float[] initialDepths, float[] initialQ) {
        log.info("Inicializando sesión GPU Manning (Lazy) para {} celdas...", cellCount);

        float[] slopeArray = calculateAndSanitizeBedSlopes(this.geometry);

        // Buffers temporales (solo viven durante esta llamada)
        FloatBuffer widthBuf   = createDirectBuffer(this.geometry.getBottomWidth());
        FloatBuffer sideBuf    = createDirectBuffer(this.geometry.getSideSlope());
        FloatBuffer manningBuf = createDirectBuffer(this.geometry.getManningCoefficient());
        FloatBuffer bedBuf     = createDirectBuffer(slopeArray);

        FloatBuffer depthBuf   = createDirectBuffer(sanitizeDepths(initialDepths));
        FloatBuffer qBuf       = createDirectBuffer(initialQ);

        this.sessionHandle = nativeSolver.initSession(
                widthBuf, sideBuf, manningBuf, bedBuf,
                depthBuf, qBuf,
                cellCount
        );

        if (this.sessionHandle == 0) {
            throw new RuntimeException("Fallo crítico al inicializar la sesión GPU de Manning.");
        }
        log.info("Sesión GPU Manning inicializada correctamente (Handle: {}).", sessionHandle);
    }

    /**
     * Resuelve un lote de pasos de tiempo en la GPU usando DMA.
     *
     * @return [batchSize][2][activeWidth] Matriz triangular/cuadrada con los resultados.
     */
    public float[][][] solveBatch(float[] initialDepths, float[] newInflows, float[] initialRiverStateQ) {
        // 1. Lazy Initialization
        if (sessionHandle == 0) {
            initializeSession(initialDepths, initialRiverStateQ);
        }

        int batchSize = newInflows.length;

        // 2. Sanitización (Ligera)
        float[] safeInflows = sanitizeInflows(newInflows);

        // 3. Preparación de Memoria DMA
        // Calculamos el tamaño exacto de salida esperado (Triangular/Cuadrado optimizado)
        int activeWidth = Math.min(batchSize, cellCount);
        int neededOutputFloats = batchSize * activeWidth * 2; // H y V

        // Redimensionamos los buffers directos si el batch creció
        ensureBuffersCapacity(batchSize, neededOutputFloats);

        // 4. Llenado del Buffer de Entrada (Java Heap -> Pinned Memory)
        this.inputBuffer.clear(); // Resetear punteros
        this.inputBuffer.put(safeInflows);
        // No es necesario 'flip' para JNI GetDirectBufferAddress, pero la posición queda al final.

        // 5. EJECUCIÓN NATIVA (Zero-Copy)
        // Pasamos los buffers, no arrays. C++ escribe directo en outputBuffer.
        int status = nativeSolver.runBatch(sessionHandle, inputBuffer, outputBuffer, batchSize);

        if (status != 0) {
            throw new RuntimeException("Error nativo en Manning GPU. Código: " + status);
        }

        // 6. Recuperación de Resultados (Pinned Memory -> Java Heap)
        // Volcamos el buffer directo a un array temporal Java para procesarlo rápido.
        // Usamos 'bulk get' que es mucho más rápido que leer float a float.
        float[] tempResults = new float[neededOutputFloats];

        this.outputBuffer.clear(); // IMPORTANTE: Resetear posición a 0 para leer desde el principio
        this.outputBuffer.get(tempResults); // Copia rápida de memoria

        // 7. Desempaquetado
        return unpackGpuResults(tempResults, batchSize);
    }

    /**
     * Asegura que los buffers tengan el tamaño correcto aplicando la lógica de histéresis:
     * - Crece si falta espacio.
     * - Se encoge si sobra más del doble de lo necesario.
     * - En ambos casos de re-alloc, aplica un factor de seguridad del 20% (x1.2).
     */
    private void ensureBuffersCapacity(int requiredInputFloats, int requiredOutputFloats) {
        this.inputBuffer = manageBufferResize(this.inputBuffer, requiredInputFloats);
        this.outputBuffer = manageBufferResize(this.outputBuffer, requiredOutputFloats);
    }

    /**
     * Implementa la lógica de resizing C++:
     * if (needed > cap || cap > needed * 2) -> reallocate(needed * 1.2)
     */
    private FloatBuffer manageBufferResize(FloatBuffer currentBuffer, int neededElements) {
        // Caso 1: Inicialización
        if (currentBuffer == null) {
            return allocateDirectFloatBuffer((int) (neededElements * 1.2f));
        }

        int currentCap = currentBuffer.capacity();

        // Lógica de Resizing (Grow or Shrink)
        // Si no cabe, O si es monstruosamente grande para lo que necesitamos hoy
        if (neededElements > currentCap || currentCap > (neededElements * 2)) {
            // Calculamos nuevo tamaño con 20% de margen de seguridad
            int newSize = (int) (neededElements * 1.2f);

            // Nota: Al perder la referencia 'currentBuffer', el Cleaner de Java
            // eventualmente liberará la memoria nativa antigua.
            return allocateDirectFloatBuffer(newSize);
        }

        // El buffer actual es válido y eficiente
        return currentBuffer;
    }

    private FloatBuffer allocateDirectFloatBuffer(int floats) {
        // Aseguramos al menos 1 float para evitar buffers de tamaño 0
        if (floats < 1) floats = 1;

        return ByteBuffer.allocateDirect(floats * 4)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer();
    }

    @Override
    public void close() {
        if (sessionHandle != 0) {
            log.info("Cerrando sesión GPU Manning (Handle: {})...", sessionHandle);
            nativeSolver.destroySession(sessionHandle);
            sessionHandle = 0;
        }
        // Ayudamos al GC liberando referencias fuertes a los buffers directos
        inputBuffer = null;
        outputBuffer = null;
    }

    // --- Helpers de Desempaquetado y Sanitización (Inalterados en lógica) ---

    private float[][][] unpackGpuResults(float[] gpuResults, int batchSize) {
        int activeWidth = Math.min(batchSize, cellCount);
        int expectedSize = batchSize * activeWidth * 2;

        if (gpuResults == null || gpuResults.length != expectedSize) {
            throw new IllegalArgumentException(String.format(
                    "Error GPU: Tamaño de resultados incorrecto. Esperado: %d, Recibido: %d",
                    expectedSize, gpuResults != null ? gpuResults.length : 0));
        }

        float[][][] results = new float[batchSize][2][activeWidth];

        // Layout GPU SoA: [ Bloque H ] [ Bloque V ]
        int offsetH = 0;
        int offsetV = batchSize * activeWidth; // Mitad del array

        int ptrH = offsetH;
        int ptrV = offsetV;

        for (int t = 0; t < batchSize; t++) {
            for (int c = 0; c < activeWidth; c++) {
                results[t][0][c] = gpuResults[ptrH++]; // H
                results[t][1][c] = gpuResults[ptrV++]; // V
            }
        }
        return results;
    }

    private float[] sanitizeInflows(float[] inflows) {
        boolean dirty = false;
        for(float f : inflows) if(f <= 0) { dirty = true; break; }
        if(!dirty) return inflows;
        float[] clean = new float[inflows.length];
        for(int i=0; i<inflows.length; i++) clean[i] = Math.max(0.001f, inflows[i]);
        return clean;
    }

    private float[] sanitizeDepths(float[] depths) {
        boolean dirty = false;
        for(float f : depths) if(f < 1e-3f) { dirty = true; break; }
        if(!dirty) return depths;
        float[] clean = new float[depths.length];
        for(int i=0; i<depths.length; i++) clean[i] = Math.max(0.001f, depths[i]);
        return clean;
    }

    private float[] calculateAndSanitizeBedSlopes(RiverGeometry geometry) {
        float[] elevations = geometry.cloneElevationProfile();
        float dx = geometry.getSpatialResolution();
        int n = geometry.getCellCount();
        float[] slopes = new float[n];
        for (int i = 0; i < n - 1; i++) {
            double s = (elevations[i] - elevations[i + 1]) / dx;
            slopes[i] = (float) Math.max(1e-7, s);
        }
        if (n > 0) slopes[n - 1] = (n > 1) ? slopes[n - 2] : 1e-7f;
        return slopes;
    }

    // Helper antiguo para initSession (ahora usa allocateDirectFloatBuffer para la nueva lógica)
    private FloatBuffer createDirectBuffer(float[] data) {
        FloatBuffer fb = allocateDirectFloatBuffer(data.length);
        fb.put(data);
        fb.position(0);
        return fb;
    }
}