package projectstalker.physics.jni;

import lombok.extern.slf4j.Slf4j;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;
import projectstalker.physics.model.FlowProfileModel;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

/**
 * Actúa como un puente (wrapper) Stateful hacia la librería nativa JNI para resolver
 * la ecuación de Manning en la GPU.
 * <p>
 * Implementa el patrón RAII: Inicializa la sesión GPU en el constructor y la libera en close().
 */
@Slf4j
public final class ManningGpuSolver implements AutoCloseable {

    private final INativeManningSolver nativeSolver;
    private long sessionHandle = 0; // Puntero a la sesión C++ (0 = inválido)
    private final int cellCount; // Guardado para validaciones

    // Constructor Principal (Stateful)
    // Se asume que este Solver se crea UNA vez por simulación y se reutiliza.
    public ManningGpuSolver(RiverGeometry geometry) {
        this(NativeManningGpuSingleton.getInstance(), geometry);
    }

    // Constructor para Inyección de Dependencias (Testing)
    public ManningGpuSolver(INativeManningSolver nativeSolver, RiverGeometry geometry) {
        this.nativeSolver = nativeSolver;
        this.cellCount = geometry.getCellCount();
        initializeSession(geometry);
    }

    /**
     * Inicializa la sesión GPU.
     * Convierte la geometría Java a Buffers Directos y llama al Init nativo.
     */
    private void initializeSession(RiverGeometry geometry) {
        log.info("Inicializando sesión GPU Manning para {} celdas...", cellCount);

        // 1. Preparar Geometría (Calculando pendientes si es necesario)
        float[] slopeArray = calculateAndSanitizeBedSlopes(geometry);

        // 2. Crear Direct Buffers (Solo viven durante esta llamada, luego C++ copia lo que necesita)
        // TODO hacer que RiverGeometry implemente FloatBuffers de forma nativa sin necesitar copia
        FloatBuffer widthBuf   = createDirectBuffer(geometry.getBottomWidth());
        FloatBuffer sideBuf    = createDirectBuffer(geometry.getSideSlope());
        FloatBuffer manningBuf = createDirectBuffer(geometry.getManningCoefficient());
        FloatBuffer bedBuf     = createDirectBuffer(slopeArray);

        // 3. Llamada Nativa (Baking)
        this.sessionHandle = nativeSolver.initSession(widthBuf, sideBuf, manningBuf, bedBuf, cellCount);

        if (this.sessionHandle == 0) {
            throw new RuntimeException("Fallo crítico al inicializar la sesión GPU de Manning.");
        }
        log.info("Sesión GPU Manning inicializada correctamente (Handle: {}).", sessionHandle);
    }

    /**
     * Resuelve un lote de pasos de tiempo completo en la GPU utilizando la sesión activa.
     *
     * @param initialDepths        Estado de profundidad en t=0 del batch (Semilla para Newton-Raphson).
     * @param newInflows           Array 1D [BatchSize] con los caudales que entran al río en cada paso t.
     * @param initialRiverStateQ   Estado de caudal en todo el río en t=0 del batch.
     * @return [batchSize][2][cellCount] donde [0] es profundidad y [1] es velocidad.
     */
    public float[][][] solveBatch(float[] initialDepths, float[] newInflows, float[] initialRiverStateQ) {
        if (sessionHandle == 0) {
            throw new IllegalStateException("Intento de usar ManningGpuSolver después de haber sido cerrado (close).");
        }

        int batchSize = newInflows.length;

        // 1. Sanitización ligera (solo de inputs crudos)
        float[] safeInflows = sanitizeInflows(newInflows);
        float[] safeInitialDepths = sanitizeDepths(initialDepths);

        // 2. Ejecución Nativa (Zero-Copy Pinning)
        // Pasamos arrays primitivos, JNI se encarga del acceso rápido.
        float[] packedResults = nativeSolver.runBatch(sessionHandle, safeInflows, safeInitialDepths, initialRiverStateQ);

        // 3. Desempaquetado (De plano a Estructurado)
        return unpackGpuResults(packedResults, batchSize, cellCount);
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

    private float[][][] unpackGpuResults(float[] gpuResults, int batchSize, int cellCount) {
        int expectedSize = batchSize * cellCount * 2;
        if (gpuResults == null || gpuResults.length != expectedSize) {
            throw new IllegalArgumentException(String.format("Error GPU: Tamaño de resultados incorrecto. Esperado: %d, Recibido: %d", expectedSize, gpuResults != null ? gpuResults.length : 0));
        }

        float[][][] results = new float[batchSize][2][cellCount];

        // Layout GPU: [Step0_Cell0_H, Step0_Cell0_V, Step0_Cell1_H, ...]
        // Es un array plano continuo.
        int ptr = 0;
        for (int t = 0; t < batchSize; t++) {
            for (int c = 0; c < cellCount; c++) {
                results[t][0][c] = gpuResults[ptr++]; // H
                results[t][1][c] = gpuResults[ptr++]; // V
            }
        }
        return results;
    }

    private float[] sanitizeInflows(float[] inflows) {
        // Validación básica: Caudales negativos -> 0.001
        boolean dirty = false;
        for(float f : inflows) if(f <= 0) { dirty = true; break; }

        if(!dirty) return inflows;

        float[] clean = new float[inflows.length];
        for(int i=0; i<inflows.length; i++) clean[i] = Math.max(0.001f, inflows[i]);
        return clean;
    }

    private float[] sanitizeDepths(float[] depths) {
        // Validación básica: Profundidad ~0 -> 0.001
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
        ByteBuffer bb = ByteBuffer.allocateDirect(data.length * 4);
        bb.order(ByteOrder.nativeOrder());
        FloatBuffer fb = bb.asFloatBuffer();
        fb.put(data);
        fb.position(0);
        return fb;
    }
}