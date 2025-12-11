package projectstalker.physics.simulator;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverSectionType;
import projectstalker.domain.river.RiverState;
import projectstalker.physics.solver.TransportSolver;
import projectstalker.physics.solver.impl.CpuFusedTransportSolver; // <--- NUEVO IMPORT
import projectstalker.physics.solver.impl.SplitOperatorTransportSolver;
import projectstalker.physics.jni.GpuMusclTransportSolver;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.fail;

@Tag("GPU")
@Slf4j
class TransportGpuAccuracyTest {

    private TransportSolver cpuSplitSolver; // Solver Clásico (Secuencial)
    private TransportSolver cpuFusedSolver; // Solver Referencia (Simultáneo)
    private TransportSolver gpuSolver;

    private final int CELL_COUNT = 50;
    private final float DX = 10.0f;
    // Tolerancia un poco más laxa para difusión compleja float vs double
    private final float EPSILON = 1e-3f;

    @BeforeEach
    void setUp() {
        // 1. Inicializar Solvers CPU
        // Usamos Split para casos generales y Fused para validar la GPU en casos mixtos
        this.cpuSplitSolver = new SplitOperatorTransportSolver();
        this.cpuFusedSolver = new CpuFusedTransportSolver();

        // 2. Inicializar Solver GPU
        try {
            this.gpuSolver = new GpuMusclTransportSolver();
        } catch (UnsatisfiedLinkError e) {
            fail("Error librería nativa: " + e.getMessage());
        }
    }

    // --- TESTS DE PARIDAD ---

    @Test
    @DisplayName("Paridad: Advección Pura (u > 0, alpha = 0, k = 0)")
    void compare_pure_advection() {
        // En advección pura, Split y Fused dan lo mismo matemáticas.
        // Usamos Split como referencia estándar.
        RiverGeometry geometry = createGeometry(0.0f, 0.0f);
        RiverState state = createSquareWaveState(1.0f);

        // dt=5s (0.5 celdas de movimiento)
        runParityTest(state, geometry, 5.0f, "Advección", cpuSplitSolver);
    }

    @Test
    @DisplayName("Paridad: Reacción Pura (u = 0, alpha = 0, k > 0)")
    void compare_pure_reaction() {
        // En reacción pura, el movimiento es 0, así que Fused/Split son idénticos.
        RiverGeometry geometry = createGeometry(0.0f, 0.1f); // k=0.1
        RiverState state = createSquareWaveState(0.0f);      // u=0

        runParityTest(state, geometry, 2.0f, "Reacción", cpuSplitSolver);
    }

    @Test
    @DisplayName("Paridad: Difusión Activa (u > 0, alpha > 0, k = 0)")
    void compare_pure_diffusion() {
        log.info(">>> TEST: Difusión/Dispersión Activa <<<");

        // 1. Geometría con dispersión alta
        RiverGeometry geometry = createGeometry(5.0f, 0.0f);

        // 2. Estado: Onda cuadrada moviéndose
        RiverState state = createSquareWaveState(1.0f);

        // 3. EJECUCIÓN Y COMPARACIÓN
        // CLAVE: Usamos 'cpuFusedSolver' porque la GPU suma (Adv + Diff) en un paso.
        // Si usáramos Split, tendríamos el error de operador (splitting error) y fallaría.
        runParityTest(state, geometry, 2.0f, "Difusión (Fused)", cpuFusedSolver);
    }

    // --- HELPERS ---

    private void runParityTest(RiverState initialState, RiverGeometry geometry, float dt, String testName, TransportSolver cpuReference) {
        log.info("[{}] Ejecutando CPU ({}) ...", testName, cpuReference.getSolverName());
        long t1 = System.nanoTime();
        RiverState resCpu = cpuReference.solve(initialState, geometry, dt);
        long tCpu = System.nanoTime() - t1;

        log.info("[{}] Ejecutando GPU...", testName);
        long t2 = System.nanoTime();
        RiverState resGpu = gpuSolver.solve(initialState, geometry, dt);
        long tGpu = System.nanoTime() - t2;

        log.info("[{}] Tiempos -> CPU: {} us | GPU: {} us", testName, tCpu/1000, tGpu/1000);

        compareStates(resCpu, resGpu);
    }

    private void compareStates(RiverState cpu, RiverState gpu) {
        float[] cCpu = cpu.contaminantConcentration();
        float[] cGpu = gpu.contaminantConcentration();

        double maxDiff = 0.0;

        // Ignoramos bordes (0 y N-1) para evitar ruido de condiciones de frontera
        for (int i = 1; i < CELL_COUNT - 1; i++) {
            double diff = Math.abs(cCpu[i] - cGpu[i]);
            maxDiff = Math.max(maxDiff, diff);

            if (diff > EPSILON) {
                log.error("Divergencia Celda {}: CPU={} vs GPU={}", i, cCpu[i], cGpu[i]);
                log.error("  CPU Context: [{}, {}, {}]", cCpu[i-1], cCpu[i], cCpu[i+1]);
                log.error("  GPU Context: [{}, {}, {}]", cGpu[i-1], cGpu[i], cGpu[i+1]);

                fail(String.format("Divergencia crítica en celda %d. Diff=%.4f", i, diff));
            }
        }
        log.info("Máxima diferencia (Interior): {}", maxDiff);
        log.info("¡PARIDAD CONFIRMADA!");
    }

    private RiverGeometry createGeometry(float alphaVal, float decayVal) {
        float[] elev = new float[CELL_COUNT];
        for(int i=0; i<CELL_COUNT; i++) elev[i] = 100.0f - i * 0.01f;

        float[] width = new float[CELL_COUNT]; Arrays.fill(width, 20.0f);
        float[] slope = new float[CELL_COUNT]; Arrays.fill(slope, 0.0f);
        float[] manning = new float[CELL_COUNT]; Arrays.fill(manning, 0.03f);
        float[] ph = new float[CELL_COUNT]; Arrays.fill(ph, 7.0f);
        RiverSectionType[] types = new RiverSectionType[CELL_COUNT];
        Arrays.fill(types, RiverSectionType.NATURAL);

        float[] alpha = new float[CELL_COUNT]; Arrays.fill(alpha, alphaVal);
        float[] decay = new float[CELL_COUNT]; Arrays.fill(decay, decayVal);

        return new RiverGeometry(
                CELL_COUNT, DX, elev, width, slope, manning,
                decay, ph, alpha, types
        );
    }

    private RiverState createSquareWaveState(float velocityVal) {
        float[] h = new float[CELL_COUNT]; Arrays.fill(h, 2.0f);
        float[] u = new float[CELL_COUNT]; Arrays.fill(u, velocityVal);
        float[] t = new float[CELL_COUNT]; Arrays.fill(t, 20.0f); // 20°C (Neutro)
        float[] ph = new float[CELL_COUNT]; Arrays.fill(ph, 7.0f);

        float[] c = new float[CELL_COUNT];
        int start = CELL_COUNT / 2 - 5;
        for(int i=0; i<10; i++) c[start + i] = 100.0f;

        return RiverState.builder()
                .waterDepth(h)
                .velocity(u)
                .contaminantConcentration(c)
                .temperature(t)
                .ph(ph)
                .build();
    }
}