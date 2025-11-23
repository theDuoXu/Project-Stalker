package projectstalker.physics.simulator;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.*;
import projectstalker.config.RiverConfig;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;
import projectstalker.factory.RiverGeometryFactory;
import projectstalker.physics.i.ITransportSolver;
import projectstalker.physics.impl.SplitOperatorTransportSolver;
import projectstalker.physics.jni.GpuMusclTransportSolver;

import java.util.Arrays;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

@Tag("GPU")
@Slf4j
@Timeout(value = 30, unit = TimeUnit.SECONDS)
class TransportGpuAccuracyTest {

    private ITransportSolver cpuSolver;
    private GpuMusclTransportSolver gpuSolver; // Tipo concreto para acceder a métodos si fuera necesario
    private RiverGeometry geometry;
    private int cellCount;

    private final float EPSILON = 1e-3f;

    @BeforeEach
    void setUp() {
        log.info("--- SETUP INICIO ---");
        RiverConfig config = RiverConfig.builder()
                .totalLength(500) // 500m
                .spatialResolution(10.0f) // 10m
                .baseWidth(20.0f)
                .averageSlope(0.001f)
                .baseManning(0.03f)
                .baseDecayRateAt20C(0.0f)
                .baseDispersionAlpha(0.0f)
                .build();

        RiverGeometryFactory factory = new RiverGeometryFactory();
        this.geometry = factory.createRealisticRiver(config);
        this.cellCount = geometry.getCellCount();
        log.info("Geometría creada. Celdas: {}", cellCount);

        if (this.geometry.getSpatial_resolution() <= 0.001) {
            throw new IllegalStateException("¡ERROR CRÍTICO! La geometría se creó con resolución 0. Revisa RiverConfig/Factory.");
        }
        this.cpuSolver = new SplitOperatorTransportSolver();
        log.info("Solver CPU instanciado.");

        try {
            this.gpuSolver = new GpuMusclTransportSolver();
            log.info("Solver GPU instanciado (Librería cargada).");
        } catch (UnsatisfiedLinkError e) {
            fail("No se pudo cargar la librería GPU: " + e.getMessage());
        }
        log.info("--- SETUP FIN ---");
    }

    @Test
    @DisplayName("Paridad: Advección Pura de Onda Cuadrada (CPU vs GPU)")
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void compare_advection_squareWave() {
        log.info(">>> INICIO TEST PARIDAD <<<");

        // 1. PREPARAR DATOS
        float[] h = new float[cellCount]; Arrays.fill(h, 2.0f);
        float[] u = new float[cellCount]; Arrays.fill(u, 1.0f);
        float[] t = new float[cellCount]; Arrays.fill(t, 20.0f);
        float[] ph = new float[cellCount]; Arrays.fill(ph, 7.0f);

        float[] c = new float[cellCount];
        int start = cellCount / 2 - 5;
        for(int i=0; i<10; i++) c[start + i] = 100.0f;

        RiverState initialState = new RiverState(h, u, c, t, ph);
        float dt = 5.0f;

        // 2. EJECUTAR CPU
        log.info("Ejecutando CPU...");
        long t1 = System.nanoTime();
        RiverState resCpu = cpuSolver.solve(initialState, geometry, dt);
        long tCpu = System.nanoTime() - t1;
        log.info("CPU OK. Tiempo: {} us", tCpu/1000);

        // 3. EJECUTAR GPU (Con logging agresivo)
        log.info("Ejecutando GPU...");

        // Pre-Validación de tamaños (Para detectar Segfaults potenciales)
        if (h.length != cellCount) log.error("FATAL: Array H length mismatch");

        long t2 = System.nanoTime();
        RiverState resGpu = null;
        try {
            resGpu = gpuSolver.solve(initialState, geometry, dt);
        } catch (Exception e) {
            log.error("EXCEPCIÓN EN GPU SOLVE:", e);
            fail("Excepción Java durante llamada GPU: " + e.getMessage());
        }
        long tGpu = System.nanoTime() - t2;
        log.info("GPU OK. Tiempo: {} us", tGpu/1000);

        // 4. VALIDACIÓN
        assertNotNull(resGpu, "El resultado GPU es nulo (posible crash nativo silencioso).");
        compareStates(resCpu, resGpu);
    }

    private void compareStates(RiverState cpu, RiverState gpu) {
        log.info("Comparando resultados...");
        float[] cCpu = cpu.contaminantConcentration();
        float[] cGpu = gpu.contaminantConcentration();

        double maxDiff = 0.0;
        for (int i = 0; i < cellCount; i++) {
            double diff = Math.abs(cCpu[i] - cGpu[i]);
            if (diff > maxDiff) maxDiff = diff;

            if (Double.isNaN(diff)) {
                fail("Detectado NaN en celda " + i + ". CPU=" + cCpu[i] + ", GPU=" + cGpu[i]);
            }

            if (diff > EPSILON) {
                // Imprimir contexto para debug
                log.error("Divergencia Celda {}: CPU={} vs GPU={}", i, cCpu[i], cGpu[i]);
                // Mostramos vecinos para ver si es un desfase de índice
                if (i > 0) log.error("  Vecino izq ({}): CPU={} vs GPU={}", i-1, cCpu[i-1], cGpu[i-1]);

                fail(String.format("Divergencia en celda %d. Diff=%.4f", i, diff));
            }
        }
        log.info("Máxima diferencia encontrada: {}", maxDiff);
        log.info("¡PARIDAD CONFIRMADA!");
    }
}