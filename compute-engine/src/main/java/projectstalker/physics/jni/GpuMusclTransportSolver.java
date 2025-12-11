package projectstalker.physics.jni;

import lombok.extern.slf4j.Slf4j;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;
import projectstalker.physics.solver.TransportSolver;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

/**
 * Implementación acelerada por GPU del solver de transporte.
 * <p>
 * Utiliza JNI y CUDA para resolver la ecuación de Advección-Difusión-Reacción.
 * Gestiona una caché de Direct Buffers para minimizar el overhead de transferencia de memoria.
 */
@Slf4j
public class GpuMusclTransportSolver implements TransportSolver {

    private final INativeTransportSolver nativeSolver;
    private double cflSafetyFactor = 0.9; // Mismo factor que en CPU

    // --- CACHÉ DE BUFFERS (Reutilización para Zero-Copy) ---
    // Nota: Estos buffers viven fuera del Heap de Java (Off-Heap)
    private FloatBuffer cachedBottomWidth;
    private FloatBuffer cachedSideSlope;
    private FloatBuffer cachedManning;
    private FloatBuffer cachedBedSlope;
    private FloatBuffer cachedAlpha;
    private FloatBuffer cachedDecay;

    private int geometryHash = -1; // Para detectar cambios de río

    public GpuMusclTransportSolver() {
        this(NativeTransportGpuSingleton.getInstance());
    }

    public GpuMusclTransportSolver(INativeTransportSolver nativeSolver) {
        this.nativeSolver = nativeSolver;
    }

    @Override
    public String getSolverName() {
        return "GPU_MUSCL_FusedKernel";
    }

    @Override
    public RiverState solve(RiverState currentState, RiverGeometry geometry, float dtGlobal) {
        // 1. Inicializar o Actualizar Caché de Geometría
        updateGeometryCacheIfNecessary(geometry);

        // 2. Preparar Buffers Dinámicos (Estado t)
        // Estos cambian en cada frame, así que los creamos "on the fly".
        // TODO OPTIMIZACIÓN FUTURA: Tener un pool de buffers reutilizables también para esto.
        int n = geometry.getCellCount();

        FloatBuffer cInBuf = createDirectBuffer(currentState.contaminantConcentration());
        FloatBuffer uBuf   = createDirectBuffer(currentState.velocity());
        FloatBuffer hBuf   = createDirectBuffer(currentState.waterDepth());
        FloatBuffer tempBuf = createDirectBuffer(currentState.temperature());

        // Calcular Áreas en Java (barato) y pasarlas
        // (Podríamos calcularlas en el kernel, pero Manning ya las usa, así que es redundante.
        // Por simplicidad aquí las recalculamos para pasar un buffer puro).
        float[] areas = new float[n];
        for (int i = 0; i < n; i++) {
            areas[i] = (float) geometry.getCrossSectionalArea(i, currentState.getWaterDepthAt(i));
        }
        FloatBuffer areaBuf = createDirectBuffer(areas);

        // 3. Calcular Sub-stepping (Igual que en CPU)
        float maxVelocity = 0.0f;
        for (float v : currentState.velocity()) maxVelocity = Math.max(maxVelocity, Math.abs(v));
        if (maxVelocity < 1e-5f) maxVelocity = 1e-5f;

        double dx = geometry.getSpatialResolution();
        double dtMaxAdvection = (dx / (double)maxVelocity) * cflSafetyFactor;

        int numSteps = (int) Math.ceil(dtGlobal / dtMaxAdvection);
        float dtSub = (float) (dtGlobal / numSteps);

        if (numSteps > 1) {
            log.debug("GPU Sub-stepping: {} pasos de {}s.", numSteps, dtSub);
        }

        // 4. LLAMADA A CÓDIGO NATIVO
        // Pasamos todos los buffers directos. C++ obtendrá sus punteros crudos.
        float[] finalConcentration = nativeSolver.solveTransportEvolution(
                cInBuf, uBuf, hBuf, areaBuf, tempBuf,
                cachedAlpha, cachedDecay, // Geometría (cacheada)
                (float) dx, dtSub, numSteps, n
        );

        // 5. Devolver Nuevo Estado
        return currentState.withContaminantConcentration(finalConcentration);
    }

    /**
     * Verifica si la geometría ha cambiado y regenera los buffers estáticos si es necesario.
     */
    private void updateGeometryCacheIfNecessary(RiverGeometry geometry) {
        if (this.geometryHash == geometry.hashCode() && cachedAlpha != null) {
            return; // Caché válida
        }

        log.info("Generando caché de geometría GPU (Direct Buffers)...");

        // Creamos Buffers Directos para todos los arrays estáticos
        // Asumimos que RiverGeometry tiene getters que devuelven float[] o double[]
        // Si devuelven double[], hacemos cast aquí.

        // Nota: Para el transporte solo necesitamos Alpha y Decay explícitamente como arrays
        // (A y U vienen del estado, dx es escalar).
        // Pero el kernel también usa BedSlope indirectamente (o Manning)?
        // Revisa 'transport_kernel.cu': NO, el kernel SOLO usa alpha y decay de la geometría.
        // U, H, A vienen del estado dinámico.

        this.cachedAlpha = createDirectBuffer(geometry.getDispersionAlpha()); // Asumiendo getter float[] o cast
        this.cachedDecay = createDirectBuffer(geometry.getBaseDecayCoefficientAt20C()); // Asumiendo getter float[] o cast

        this.geometryHash = geometry.hashCode();
    }

    /**
     * Helper para crear Direct FloatBuffer desde float[].
     */
    private FloatBuffer createDirectBuffer(float[] data) {
        ByteBuffer bb = ByteBuffer.allocateDirect(data.length * 4);
        bb.order(ByteOrder.nativeOrder());
        FloatBuffer fb = bb.asFloatBuffer();
        fb.put(data);
        fb.position(0);
        return fb;
    }

    // Sobrecarga para double[] (convierte a float on-the-fly)
    private FloatBuffer createDirectBuffer(double[] data) {
        ByteBuffer bb = ByteBuffer.allocateDirect(data.length * 4);
        bb.order(ByteOrder.nativeOrder());
        FloatBuffer fb = bb.asFloatBuffer();
        for (double d : data) {
            fb.put((float) d);
        }
        fb.position(0);
        return fb;
    }
}