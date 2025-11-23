package projectstalker.physics.simulator;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverSectionType;
import projectstalker.domain.river.RiverState;
import projectstalker.physics.i.ITransportSolver;
import projectstalker.physics.impl.SplitOperatorTransportSolver;
import projectstalker.physics.jni.GpuMusclTransportSolver;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.fail;

@Tag("GPU")
@Slf4j
class TransportGpuAccuracyTest {

    private ITransportSolver cpuSolver;
    private ITransportSolver gpuSolver;

    private final int CELL_COUNT = 50;
    private final float DX = 10.0f;
    private final float EPSILON = 1e-3f;

    @BeforeEach
    void setUp() {
        // Inicializamos los solvers una vez (son stateless respecto a la geometría)
        // Forzamos CFL 0.9 en CPU para evitar problemas numéricos
        this.cpuSolver = new SplitOperatorTransportSolver(
                new projectstalker.physics.impl.MusclAdvectionSolver(),
                new projectstalker.physics.impl.CentralDiffusionSolver(),
                new projectstalker.physics.impl.FirstOrderReactionSolver(),
                0.9
        );

        try {
            this.gpuSolver = new GpuMusclTransportSolver();
        } catch (UnsatisfiedLinkError e) {
            fail("No se pudo cargar la librería GPU: " + e.getMessage());
        }
    }

    // --- TESTS ESPECÍFICOS ---

    @Test
    @DisplayName("Paridad: Advección Pura (u > 0, alpha = 0, k = 0)")
    void compare_pure_advection() {
        log.info(">>> TEST: Advección Pura <<<");

        // 1. Geometría sin dispersión ni reacción
        RiverGeometry geometry = createGeometry(0.0f, 0.0f);

        // 2. Estado: Onda cuadrada, velocidad constante
        RiverState state = createSquareWaveState(1.0f); // u = 1 m/s

        // 3. Ejecutar y Comparar
        // dt = 5s. La onda se mueve 5m (0.5 celdas).
        runParityTest(state, geometry, 5.0f, "Advección");
    }

    @Test
    @DisplayName("Paridad: Reacción Pura (u = 0, alpha = 0, k > 0)")
    void compare_pure_reaction() {
        log.info(">>> TEST: Reacción Pura (Decaimiento en sitio) <<<");

        // 1. Geometría con decaimiento fuerte
        // k = 0.1 (10% por segundo aprox)
        RiverGeometry geometry = createGeometry(0.0f, 0.1f);

        // 2. Estado: Onda cuadrada, AGUA QUIETA (u = 0)
        // Al ser u=0, la advección y la dispersión de Taylor se anulan.
        RiverState state = createSquareWaveState(0.0f);

        // 3. Ejecutar y Comparar
        // dt = 2s. El pico de 100 debería bajar a 100 * exp(-0.1 * 2) ~= 81.87
        runParityTest(state, geometry, 2.0f, "Reacción");
    }

    @Test
    @DisplayName("Paridad: Difusión Activa (u > 0, alpha > 0, k = 0)")
    void compare_pure_diffusion() {
        log.info(">>> TEST: Difusión/Dispersión Activa <<<");
        log.warn("Nota: En este modelo, la difusión depende de la velocidad (Taylor).");
        log.warn("Se prueba Advección + Difusión simultánea.");

        // 1. Geometría con dispersión muy alta
        // Alpha = 5.0. D_L = 5.0 * 1.0 * 2.0 = 10 m²/s. ¡Mucha mezcla!
        RiverGeometry geometry = createGeometry(5.0f, 0.0f);

        // 2. Estado: Onda cuadrada, velocidad constante
        RiverState state = createSquareWaveState(1.0f);

        // 3. Ejecutar y Comparar
        // dt = 2s. La onda se moverá poco (2m) pero se ensanchará mucho.
        runParityTest(state, geometry, 2.0f, "Difusión");
    }

    // --- HELPERS ---

    private void runParityTest(RiverState initialState, RiverGeometry geometry, float dt, String testName) {
        log.info("[{}] Ejecutando CPU...", testName);
        long t1 = System.nanoTime();
        RiverState resCpu = cpuSolver.solve(initialState, geometry, dt);
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

        // Ignoramos bordes para evitar ruido de condiciones de frontera numéricas
        for (int i = 1; i < CELL_COUNT - 1; i++) {
            double diff = Math.abs(cCpu[i] - cGpu[i]);
            maxDiff = Math.max(maxDiff, diff);

            if (diff > EPSILON) {
                log.error("Divergencia Celda {}: CPU={} vs GPU={}", i, cCpu[i], cGpu[i]);
                // Contexto visual
                log.error("  CPU Context: [{}, {}, {}]", cCpu[i-1], cCpu[i], cCpu[i+1]);
                log.error("  GPU Context: [{}, {}, {}]", cGpu[i-1], cGpu[i], cGpu[i+1]);

                fail(String.format("Divergencia crítica. Diff=%.4f", diff));
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

        // Arrays Específicos del Test
        float[] alpha = new float[CELL_COUNT]; Arrays.fill(alpha, alphaVal);
        float[] decay = new float[CELL_COUNT]; Arrays.fill(decay, decayVal);

        return new RiverGeometry(
                CELL_COUNT, DX, elev, width, slope, manning,
                decay, ph, alpha, types
        );
    }

    private RiverState createSquareWaveState(float velocityVal) {
        float[] h = new float[CELL_COUNT]; Arrays.fill(h, 2.0f); // Profundidad 2m
        float[] u = new float[CELL_COUNT]; Arrays.fill(u, velocityVal);
        float[] t = new float[CELL_COUNT]; Arrays.fill(t, 20.0f); // 20°C (Neutro para Arrhenius)
        float[] ph = new float[CELL_COUNT]; Arrays.fill(ph, 7.0f);

        float[] c = new float[CELL_COUNT];
        // Onda cuadrada en el centro
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