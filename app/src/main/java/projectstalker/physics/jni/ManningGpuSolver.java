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
 * 2. Full Evolution: Robusto para ciencia (matriz completa con Stride). Sin restricciones.
 * <p>
 * MODIFICADO: Ahora devuelve resultados crudos (Raw Arrays) para optimizar el paso de mensajes
 * y permitir la construcción flexible de DTOs (Strided, Chunked, Flyweight).
 */
@Slf4j
public final class ManningGpuSolver implements AutoCloseable {

    /**
     * DTO interno para transporte de datos crudos desde la GPU.
     * depths y velocities son arrays planos concatenados (Time * Cells).
     */
    public record RawGpuResult(float[] depths, float[] velocities, int storedSteps) {}

    private final INativeManningSolver nativeSolver;
    private long sessionHandle = 0;
    private final int cellCount;
    private final RiverGeometry geometry;

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

    // --- MÉTODOS PÚBLICOS ---

    /**
     * Ejecuta la estrategia SMART.
     * Devuelve los datos crudos. Stride siempre es 1.
     */
    public RawGpuResult solveSmartBatch(float[] initialDepths,
                                        float[] newInflows,
                                        float[] initialRiverStateQ,
                                        boolean trustOptimization) {
        if (!trustOptimization) {
            validateSteadyStateCondition(initialRiverStateQ);
        }

        int batchSize = newInflows.length;
        // En la nueva arquitectura, aunque Smart calcule un triángulo,
        // pedimos/manejamos el ancho completo para simplificar la estructura de arrays planos.
        // El kernel optimizado simplemente no escribirá fuera del triángulo, quedando los valores a 0 o NaN
        // (Flyweight se encarga luego de gestionar eso con la "ola").

        return executeNativeBatch(
                initialDepths, newInflows, initialRiverStateQ,
                batchSize,
                INativeManningSolver.MODE_SMART_LAZY,
                1 // Stride siempre es 1 en Smart (Triángulo denso)
        );
    }

    /**
     * Ejecuta la estrategia FULL EVOLUTION.
     * Requiere que el batchSize sea múltiplo del stride.
     */
    public RawGpuResult solveFullEvolutionBatch(float[] initialDepths,
                                                float[] newInflows,
                                                float[] initialRiverStateQ,
                                                int stride) { // <--- STRIDE EXPLICITO
        int batchSize = newInflows.length;

        // --- VALIDACIÓN DE ALINEACIÓN ---
        if (stride > 1 && batchSize % stride != 0) {
            throw new IllegalArgumentException(String.format(
                    "Error de Alineación GPU: El tamaño del lote (%d) NO es múltiplo del stride (%d). " +
                            "Esto provocaría discontinuidades temporales en la salida.",
                    batchSize, stride));
        }

        return executeNativeBatch(
                initialDepths, newInflows, initialRiverStateQ,
                batchSize,
                INativeManningSolver.MODE_FULL_EVOLUTION,
                stride
        );
    }

    // Sobrecarga por defecto (Stride = 1)
    public RawGpuResult solveFullEvolutionBatch(float[] initialDepths,
                                                float[] newInflows,
                                                float[] initialRiverStateQ) {
        return solveFullEvolutionBatch(initialDepths, newInflows, initialRiverStateQ, 1);
    }

    // --- Lógica Común ---

    private RawGpuResult executeNativeBatch(float[] initialDepths,
                                            float[] newInflows,
                                            float[] initialQ,
                                            int batchSize,
                                            int mode,
                                            int stride) { // <--- STRIDE
        if (sessionHandle == 0) {
            initializeSession(initialDepths, initialQ);
        }

        // Cálculo de pasos guardados.
        // Como validamos batchSize % stride == 0, la división es exacta.
        int savedSteps = batchSize / stride;

        // Protección para caso borde batch < stride (si ocurriera)
        if (savedSteps == 0 && batchSize > 0) savedSteps = 1;

        // Tamaño total de salida: (PasosGuardados * Celdas * 2 variables)
        int neededOutputFloats = savedSteps * cellCount * 2;

        ensureBuffersCapacity(batchSize, neededOutputFloats);

        this.inputBuffer.clear();
        this.inputBuffer.put(sanitizeInflows(newInflows));

        int status = nativeSolver.runBatch(
                sessionHandle,
                inputBuffer,
                outputBuffer,
                batchSize,
                mode,
                stride
        );

        if (status != 0) {
            throw new RuntimeException("Error nativo en Manning GPU. Código: " + status);
        }

        // Extracción de Resultados (Device -> Host)
        // Copiamos del buffer directo a un array temporal
        float[] allData = new float[neededOutputFloats];
        this.outputBuffer.clear();
        this.outputBuffer.get(allData);

        // Separación de H y V (SoA Unpacking)
        // Asumiendo layout: [Todas las H] seguido de [Todas las V]
        int elementsPerVar = savedSteps * cellCount;
        float[] depths = new float[elementsPerVar];
        float[] velocities = new float[elementsPerVar];

        System.arraycopy(allData, 0, depths, 0, elementsPerVar);
        System.arraycopy(allData, elementsPerVar, velocities, 0, elementsPerVar);

        return new RawGpuResult(depths, velocities, savedSteps);
    }

    // --- Helpers ---

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

    private void validateSteadyStateCondition(float[] q) {
        if (q == null || q.length < 6) return;
        float qStart = q[2];
        float qEnd = q[q.length - 3];
        float diff = Math.abs(qStart - qEnd);
        float maxQ = Math.max(Math.abs(qStart), Math.abs(qEnd));
        boolean isStable = (maxQ < 1.0f) ? (diff < 0.1f) : ((diff / maxQ) <= STEADY_STATE_TOLERANCE);
        if (!isStable) throw new IllegalStateException("Río inestable para optimización Smart.");
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

    @Override
    public void close() {
        if (sessionHandle != 0) {
            nativeSolver.destroySession(sessionHandle);
            sessionHandle = 0;
        }
        inputBuffer = null;
        outputBuffer = null;
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

    private FloatBuffer createDirectBuffer(float[] data) {
        FloatBuffer fb = allocateDirectFloatBuffer(data.length);
        fb.put(data);
        fb.position(0);
        return fb;
    }
}