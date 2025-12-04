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
 * Implementa el patrón RAII: Inicializa la sesión GPU de forma perezosa (Lazy) y la libera en close().
 */
@Slf4j
public final class ManningGpuSolver implements AutoCloseable {

    private final INativeManningSolver nativeSolver;
    private long sessionHandle = 0; // Puntero a la sesión C++ (0 = inválido)
    private final int cellCount; // Guardado para validaciones
    private final RiverGeometry geometry; // Guardamos referencia para el Lazy Init

    // Constructor Principal (Stateful)
    // Se asume que este Solver se crea UNA vez por simulación y se reutiliza.
    public ManningGpuSolver(RiverGeometry geometry) {
        this(NativeManningGpuSingleton.getInstance(), geometry);
    }

    // Constructor para Inyección de Dependencias (Testing)
    public ManningGpuSolver(INativeManningSolver nativeSolver, RiverGeometry geometry) {
        this.nativeSolver = nativeSolver;
        this.geometry = geometry;
        this.cellCount = geometry.getCellCount();
        // NOTA: Ya no llamamos a initializeSession aquí (Lazy Init)
    }

    /**
     * Inicializa la sesión GPU.
     * Convierte la geometría Java y el ESTADO INICIAL a Buffers Directos y llama al Init nativo.
     * Se ejecuta solo la primera vez que se llama a solveBatch.
     */
    private void initializeSession(float[] initialDepths, float[] initialQ) {
        log.info("Inicializando sesión GPU Manning (Lazy) para {} celdas...", cellCount);

        // 1. Preparar Geometría (Calculando pendientes si es necesario)
        float[] slopeArray = calculateAndSanitizeBedSlopes(this.geometry);

        // 2. Crear Direct Buffers (Geometría)
        FloatBuffer widthBuf   = createDirectBuffer(this.geometry.getBottomWidth());
        FloatBuffer sideBuf    = createDirectBuffer(this.geometry.getSideSlope());
        FloatBuffer manningBuf = createDirectBuffer(this.geometry.getManningCoefficient());
        FloatBuffer bedBuf     = createDirectBuffer(slopeArray);

        // 3. Crear Direct Buffers (Estado Inicial - Flyweight Intrinsic)
        // Sanitizamos antes de enviar para asegurar estabilidad numérica en el estado base
        FloatBuffer depthBuf   = createDirectBuffer(sanitizeDepths(initialDepths));
        FloatBuffer qBuf       = createDirectBuffer(initialQ); // Q puede ser negativo, no sanitizamos agresivamente

        // 4. Llamada Nativa (Baking + Carga)
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
     * Resuelve un lote de pasos de tiempo en la GPU.
     *
     * @param initialDepths        Estado de profundidad en t=0 del batch. Se usa para initSession la primera vez.
     * @param newInflows           Array 1D [BatchSize] con los caudales que entran al río.
     * @param initialRiverStateQ   Estado de caudal en t=0 del batch. Se usa para initSession la primera vez.
     * @return [batchSize][2][batchSize] Matriz triangular/cuadrada con los NUEVOS datos calculados.
     */
    public float[][][] solveBatch(float[] initialDepths, float[] newInflows, float[] initialRiverStateQ) {
        // --- LAZY INITIALIZATION ---
        if (sessionHandle == 0) {
            initializeSession(initialDepths, initialRiverStateQ);
        }

        int batchSize = newInflows.length;

        // 1. Sanitización ligera (solo de inputs dinámicos)
        float[] safeInflows = sanitizeInflows(newInflows);

        // 2. Ejecución Nativa (Zero-Copy Pinning)
        // Pasamos SOLO lo que cambia (Flyweight Extrinsic State)
        float[] packedResults = nativeSolver.runBatch(sessionHandle, safeInflows);

        // 3. Desempaquetado (SoA -> Estructurado 3D)
        // Devuelve la matriz cuadrada activa, no todo el río.
        return unpackGpuResults(packedResults, batchSize);
    }

    /**
     * Libera los recursos de la GPU.
     */
    @Override
    public void close() {
        if (sessionHandle != 0) {
            log.info("Cerrando sesión GPU Manning (Handle: {})...", sessionHandle);
            nativeSolver.destroySession(sessionHandle);
            sessionHandle = 0;
        }
    }

    // --- Helpers de Desempaquetado y Sanitización ---

    private float[][][] unpackGpuResults(float[] gpuResults, int batchSize) {
        // El tamaño esperado ahora es el cuadrado del batch (H y V)
        // Tamaño = BatchSize * BatchSize * 2
        int activeWidth = Math.min(batchSize, cellCount); // Seguridad por si el batch es gigante
        int expectedSize = batchSize * activeWidth * 2;

        if (gpuResults == null || gpuResults.length != expectedSize) {
            throw new IllegalArgumentException(String.format(
                    "Error GPU: Tamaño de resultados incorrecto. Esperado: %d, Recibido: %d",
                    expectedSize, gpuResults != null ? gpuResults.length : 0));
        }

        // Matriz de retorno: [Tiempo][Variable][Espacio_Activo]
        // Nota: La dimensión espacial es 'activeWidth' (BatchSize), NO 'cellCount'.
        float[][][] results = new float[batchSize][2][activeWidth];

        // Layout GPU SoA: [ Bloque H (size=NxN) | Bloque V (size=NxN) ]
        int blockSize = batchSize * activeWidth;
        int offsetH = 0;
        int offsetV = blockSize;

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
        // (Nota: Usamos clone aquí porque es una operación única en init)
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

    private FloatBuffer createDirectBuffer(float[] data) {
        ByteBuffer bb = ByteBuffer.allocateDirect(data.length * 4);
        bb.order(ByteOrder.nativeOrder());
        FloatBuffer fb = bb.asFloatBuffer();
        fb.put(data);
        fb.position(0);
        return fb;
    }
}