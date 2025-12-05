package projectstalker.physics.jni;

import lombok.extern.slf4j.Slf4j;
import projectstalker.domain.river.RiverGeometry;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

/**
 * Actúa como un puente (wrapper) Stateful hacia la librería nativa JNI.
 * Soporta dos estrategias:
 * 1. Smart/Lazy: Optimizado para alertas (recorte triangular). Requiere estado estable.
 * 2. Full Evolution: Robusto para ciencia (matriz completa). Sin restricciones.
 */
@Slf4j
public final class ManningGpuSolver implements AutoCloseable {

    private final INativeManningSolver nativeSolver;
    private long sessionHandle = 0;
    private final int cellCount;
    private final RiverGeometry geometry;

    // Tolerancia para considerar Steady State (5%)
    private static final float STEADY_STATE_TOLERANCE = 0.05f;

    // --- DMA Buffers Persistentes ---
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

    // --- MÉTODOS PÚBLICOS DE RESOLUCIÓN ---

    /**
     * Estrategia OPTIMIZADA (Smart/Lazy).
     * Calcula y transfiere solo el triángulo de influencia de la nueva ola.
     * <p>
     * REQUISITO: El río aguas abajo debe estar en equilibrio (Steady State).
     * Si no se confía en la optimización, se verificará el caudal.
     */
    public float[][][] solveSmartBatch(float[] initialDepths,
                                       float[] newInflows,
                                       float[] initialRiverStateQ,
                                       boolean trustOptimization) {
        // 1. Validación de Seguridad (Solo para modo Smart)
        if (!trustOptimization) {
            validateSteadyStateCondition(initialRiverStateQ);
        }

        // 2. Configuración de Tamaño (Triangular/Cuadrado recortado)
        int batchSize = newInflows.length;
        int activeWidth = Math.min(batchSize, cellCount);

        // 3. Ejecución
        return executeNativeBatch(
                initialDepths, newInflows, initialRiverStateQ,
                activeWidth, // Ancho de salida esperado
                INativeManningSolver.MODE_SMART_LAZY
        );
    }

    /**
     * Estrategia ROBUSTA (Full Evolution).
     * Calcula y transfiere la matriz rectangular completa (Batch x CellCount).
     * <p>
     * Úsalo cuando el río ya tiene olas complejas o el estado no es estacionario.
     * Es más lento en transferencia (PCIe), pero físicamente riguroso.
     */
    public float[][][] solveFullEvolutionBatch(float[] initialDepths,
                                               float[] newInflows,
                                               float[] initialRiverStateQ) {
        // En Full Mode siempre traemos todo el río
        int batchSize = newInflows.length;
        int fullWidth = cellCount;

        return executeNativeBatch(
                initialDepths, newInflows, initialRiverStateQ,
                fullWidth, // Ancho de salida total
                INativeManningSolver.MODE_FULL_EVOLUTION
        );
    }

    // --- Lógica Común de Ejecución (DRY) ---

    private float[][][] executeNativeBatch(float[] initialDepths,
                                           float[] newInflows,
                                           float[] initialQ,
                                           int targetOutputWidth,
                                           int mode) {
        // A. Lazy Init
        if (sessionHandle == 0) {
            initializeSession(initialDepths, initialQ);
        }

        int batchSize = newInflows.length;

        // B. Preparación de Memoria DMA
        // Calculamos el tamaño necesario según la estrategia (Target Width)
        int neededOutputFloats = batchSize * targetOutputWidth * 2; // H y V

        ensureBuffersCapacity(batchSize, neededOutputFloats);

        // C. Llenado Input (Pinned)
        this.inputBuffer.clear();
        this.inputBuffer.put(sanitizeInflows(newInflows));

        // D. Llamada Nativa (Zero-Copy)
        int status = nativeSolver.runBatch(
                sessionHandle,
                inputBuffer,
                outputBuffer,
                batchSize,
                mode // Pasamos el modo a C++
        );

        if (status != 0) {
            throw new RuntimeException("Error nativo en Manning GPU. Código: " + status + " Modo: " + mode);
        }

        // E. Recuperación Resultados
        float[] tempResults = new float[neededOutputFloats];
        this.outputBuffer.clear();
        this.outputBuffer.get(tempResults);

        // F. Desempaquetado
        return unpackGpuResults(tempResults, batchSize, targetOutputWidth);
    }

    // --- Helpers Internos ---

    private void initializeSession(float[] initialDepths, float[] initialQ) {
        log.info("Inicializando sesión GPU Manning (Lazy) para {} celdas...", cellCount);
        float[] slopeArray = calculateAndSanitizeBedSlopes(this.geometry);

        FloatBuffer widthBuf   = createDirectBuffer(this.geometry.getBottomWidth());
        FloatBuffer sideBuf    = createDirectBuffer(this.geometry.getSideSlope());
        FloatBuffer manningBuf = createDirectBuffer(this.geometry.getManningCoefficient());
        FloatBuffer bedBuf     = createDirectBuffer(slopeArray);
        FloatBuffer depthBuf   = createDirectBuffer(sanitizeDepths(initialDepths));
        FloatBuffer qBuf       = createDirectBuffer(initialQ);

        this.sessionHandle = nativeSolver.initSession(widthBuf, sideBuf, manningBuf, bedBuf, depthBuf, qBuf, cellCount);

        if (this.sessionHandle == 0) throw new RuntimeException("Fallo crítico al inicializar sesión GPU.");
        log.info("Sesión GPU inicializada (Handle: {}).", sessionHandle);
    }

    /**
     * Verifica la estabilidad del caudal base.
     * Corrección aplicada: Índices [2] y [length-3] para evitar efectos de borde.
     */
    private void validateSteadyStateCondition(float[] q) {
        if (q == null || q.length < 6) return; // Array muy pequeño, no validamos o asumimos OK.

        // Muestreo evitando las primerísimas celdas (condición de contorno)
        float qStart = q[2];
        float qEnd = q[q.length - 3];

        float diff = Math.abs(qStart - qEnd);
        float maxQ = Math.max(Math.abs(qStart), Math.abs(qEnd));

        boolean isStable;
        if (maxQ < 1.0f) {
            isStable = diff < 0.1f;
        } else {
            isStable = (diff / maxQ) <= STEADY_STATE_TOLERANCE;
        }

        if (!isStable) {
            throw new IllegalStateException(String.format(
                    "OPTIMIZACIÓN INSEGURA: El caudal base no es estable (Steady State).\n" +
                            "Q[2]=%.2f, Q[N-3]=%.2f (Delta=%.1f%%).\n" +
                            "El modo Smart requiere caudal uniforme. Use 'trustOptimization=true' o 'solveFullEvolutionBatch'.",
                    qStart, qEnd, (diff / maxQ) * 100f));
        }
    }

    private void ensureBuffersCapacity(int requiredInputFloats, int requiredOutputFloats) {
        this.inputBuffer = manageBufferResize(this.inputBuffer, requiredInputFloats);
        this.outputBuffer = manageBufferResize(this.outputBuffer, requiredOutputFloats);
    }

    private FloatBuffer manageBufferResize(FloatBuffer currentBuffer, int neededElements) {
        if (currentBuffer == null) return allocateDirectFloatBuffer((int) (neededElements * 1.2f));
        int currentCap = currentBuffer.capacity();
        if (neededElements > currentCap || currentCap > (neededElements * 2)) {
            return allocateDirectFloatBuffer((int) (neededElements * 1.2f));
        }
        return currentBuffer;
    }

    private FloatBuffer allocateDirectFloatBuffer(int floats) {
        if (floats < 1) floats = 1;
        return ByteBuffer.allocateDirect(floats * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
    }

    /**
     * Desempaqueta los resultados planos de la GPU a una matriz estructurada.
     * Acepta 'activeWidth' dinámico para soportar tanto Smart (triángulo) como Full (rectángulo).
     */
    private float[][][] unpackGpuResults(float[] gpuResults, int batchSize, int activeWidth) {
        int expectedSize = batchSize * activeWidth * 2;

        if (gpuResults == null || gpuResults.length != expectedSize) {
            throw new IllegalArgumentException(String.format(
                    "Error GPU Unpack: Tamaño incorrecto. Esperado: %d, Recibido: %d (Batch=%d, Width=%d)",
                    expectedSize, (gpuResults != null ? gpuResults.length : 0), batchSize, activeWidth));
        }

        float[][][] results = new float[batchSize][2][activeWidth];

        // Layout GPU SoA: [ Bloque H (Batch x Width) ] [ Bloque V (Batch x Width) ]
        int ptrH = 0;
        int ptrV = batchSize * activeWidth; // Inicio del bloque V

        for (int t = 0; t < batchSize; t++) {
            for (int c = 0; c < activeWidth; c++) {
                results[t][0][c] = gpuResults[ptrH++];
                results[t][1][c] = gpuResults[ptrV++];
            }
        }
        return results;
    }

    @Override
    public void close() {
        if (sessionHandle != 0) {
            log.info("Cerrando sesión GPU Manning (Handle: {})...", sessionHandle);
            nativeSolver.destroySession(sessionHandle);
            sessionHandle = 0;
        }
        inputBuffer = null;
        outputBuffer = null;
    }

    // --- Helpers de Sanitización y Geometría ---
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

    private FloatBuffer createDirectBuffer(float[] data) {
        FloatBuffer fb = allocateDirectFloatBuffer(data.length);
        fb.put(data);
        fb.position(0);
        return fb;
    }
}