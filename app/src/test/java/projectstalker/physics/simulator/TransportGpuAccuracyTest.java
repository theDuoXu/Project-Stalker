package projectstalker.physics.simulator;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import projectstalker.config.RiverConfig;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;
import projectstalker.factory.RiverGeometryFactory;
import projectstalker.physics.impl.SplitOperatorTransportSolver;
import projectstalker.physics.i.ITransportSolver;
import projectstalker.physics.jni.GpuMusclTransportSolver;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

@Tag("GPU")
@Slf4j
class TransportGpuAccuracyTest {

    private ITransportSolver cpuSolver;
    private ITransportSolver gpuSolver;
    private RiverGeometry geometry;
    private int cellCount;

    // Tolerancia: 1e-3 es razonable para diferencias acumuladas float vs double
    private final float EPSILON = 1e-3f;

    @BeforeEach
    void setUp() {
        // 1. Geometría Controlada
        RiverConfig config = RiverConfig.builder()
                .totalLength(5000) // 5 km
                .spatialResolution(10.0f) // 10m
                .baseWidth(20.0f)
                .averageSlope(0.001f)
                .baseManning(0.03f)
                .baseDecayRateAt20C(0.0f) // Sin reacción inicial
                .baseDispersionAlpha(0.0f) // Sin dispersión inicial
                .build();

        RiverGeometryFactory factory = new RiverGeometryFactory();
        this.geometry = factory.createRealisticRiver(config);
        this.cellCount = geometry.getCellCount();

        // 2. Instanciar Solvers
        this.cpuSolver = new SplitOperatorTransportSolver();

        try {
            this.gpuSolver = new GpuMusclTransportSolver();
        } catch (UnsatisfiedLinkError e) {
            fail("No se pudo cargar la librería GPU. Ejecuta con ./gradlew gpuTest");
        }

        log.info("Setup completo. Celdas: {}", cellCount);
    }

    @Test
    @DisplayName("Paridad: Advección Pura de Onda Cuadrada (CPU vs GPU)")
    void compare_advection_squareWave() {
        log.info(">>> TEST: Comparativa Advección Pura");

        // ARRANGE
        float[] h = new float[cellCount]; Arrays.fill(h, 2.0f);
        float[] u = new float[cellCount]; Arrays.fill(u, 1.0f); // 1 m/s
        float[] t = new float[cellCount]; Arrays.fill(t, 20.0f);
        float[] ph = new float[cellCount]; Arrays.fill(ph, 7.0f);

        float[] c = new float[cellCount];
        // Onda cuadrada de 10 celdas en el medio
        int start = cellCount / 2;
        for(int i=0; i<10; i++) c[start + i] = 100.0f;

        RiverState initialState = new RiverState(h, u, c, t, ph);

        // Paso de tiempo: 5 segundos. (u*dt = 5m = 0.5 celdas).
        // Esto fuerza al solver MUSCL a interpolar (caso difícil).
        float dt = 5.0f;

        // ACT
        long t1 = System.nanoTime();
        RiverState resCpu = cpuSolver.solve(initialState, geometry, dt);
        long tCpu = System.nanoTime() - t1;

        long t2 = System.nanoTime();
        RiverState resGpu = gpuSolver.solve(initialState, geometry, dt);
        long tGpu = System.nanoTime() - t2;

        log.info("Tiempo CPU: {} us | Tiempo GPU: {} us", tCpu/1000, tGpu/1000);

        // ASSERT
        compareStates(resCpu, resGpu);
    }

    private void compareStates(RiverState cpu, RiverState gpu) {
        float[] cCpu = cpu.contaminantConcentration();
        float[] cGpu = gpu.contaminantConcentration();

        double maxDiff = 0.0;
        for (int i = 0; i < cellCount; i++) {
            double diff = Math.abs(cCpu[i] - cGpu[i]);
            if (diff > maxDiff) maxDiff = diff;

            if (diff > EPSILON) {
                fail(String.format("Divergencia en celda %d. CPU=%.4f, GPU=%.4f, Diff=%.4f",
                        i, cCpu[i], cGpu[i], diff));
            }
        }
        log.info("Máxima diferencia encontrada: {}", maxDiff);
        log.info("¡PARIDAD CONFIRMADA!");
    }
}