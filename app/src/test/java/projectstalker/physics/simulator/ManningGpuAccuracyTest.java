package projectstalker.physics.simulator;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import projectstalker.config.RiverConfig;
import projectstalker.config.SimulationConfig;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;
import projectstalker.domain.simulation.ISimulationResult;
import projectstalker.factory.RiverGeometryFactory;
import projectstalker.physics.impl.ManningProfileCalculatorTask;
import projectstalker.config.SimulationConfig.GpuStrategy;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

@Tag("GPU")
@Slf4j
class ManningGpuAccuracyTest {

    private RiverGeometry realGeometry;
    private SimulationConfig cpuConfig;
    private SimulationConfig gpuConfig;

    private int cellCount;
    private final int BATCH_SIZE = 5;

    // Tolerancia (Float GPU vs Double CPU)
    private final float EPSILON = 1e-2f;

    private RiverState stableInitialState;
    private final float BASE_DISCHARGE = 50.0f;

    @BeforeEach
    void setUp() throws Exception {
        RiverConfig riverConfig = RiverConfig.getTestingRiver();
        RiverGeometryFactory factory = new RiverGeometryFactory();
        this.realGeometry = factory.createRealisticRiver(riverConfig);
        this.cellCount = this.realGeometry.getCellCount();

        log.info("Test configurado. Geometría real tiene {} celdas.", this.cellCount);

        // Generar Estado Estable (CPU Warm-Up)
        float[] qProfile = new float[cellCount];
        Arrays.fill(qProfile, BASE_DISCHARGE);
        float[] seedDepth = new float[cellCount];
        Arrays.fill(seedDepth, 1.0f);

        ManningProfileCalculatorTask calculator = new ManningProfileCalculatorTask(
                qProfile, seedDepth, realGeometry
        );
        calculator.call();

        this.stableInitialState = new RiverState(
                calculator.getCalculatedWaterDepth(),
                calculator.getCalculatedVelocity(),
                new float[cellCount], new float[cellCount], new float[cellCount]
        );

        // Config
        SimulationConfig baseConfig = SimulationConfig.builder()
                .riverConfig(riverConfig)
                .seed(12345L)
                .totalTime(3600)
                .deltaTime(10.0f)
                .cpuProcessorCount(4)
                .useGpuAccelerationOnTransport(false)
                .build();

        this.cpuConfig = baseConfig.withUseGpuAccelerationOnManning(false);
        this.gpuConfig = baseConfig.withUseGpuAccelerationOnManning(true);
    }

    @Test
    @DisplayName("Paridad SMART: El kernel optimizado (triangular) debe coincidir con CPU en Steady State")
    void compareCpuVsGpu_SmartMode() {
        log.info("=== TEST PARIDAD: SMART / LAZY MODE ===");
        runComparisonTest(GpuStrategy.SMART_SAFE);
    }

    @Test
    @DisplayName("Paridad FULL: El kernel robusto (rectangular) debe coincidir con CPU")
    void compareCpuVsGpu_FullEvolutionMode() {
        log.info("=== TEST PARIDAD: FULL EVOLUTION MODE ===");
        runComparisonTest(GpuStrategy.FULL_EVOLUTION);
    }

    private void runComparisonTest(GpuStrategy strategy) {
        RiverState initialState = this.stableInitialState;

        // Input: Onda de Avenida
        float[] flowInput = new float[BATCH_SIZE];
        Arrays.fill(flowInput, 150.0f);

        float[][][] phTmp = new float[BATCH_SIZE][2][cellCount];

        ISimulationResult resultCpu;
        ISimulationResult resultGpu;

        // 1. CPU Reference
        try (ManningBatchProcessor cpuProcessor = new ManningBatchProcessor(realGeometry, cpuConfig)) {
            // CPU ignora la estrategia, pero la pasamos por firma
            resultCpu = cpuProcessor.processBatch(BATCH_SIZE, initialState, flowInput, phTmp, false, strategy);
        }

        // 2. GPU SUT
        try (ManningBatchProcessor gpuProcessor = new ManningBatchProcessor(realGeometry, gpuConfig)) {
            log.info("Ejecutando GPU con estrategia: {}", strategy);
            resultGpu = gpuProcessor.processBatch(BATCH_SIZE, initialState, flowInput, phTmp, true, strategy);
        }

        // 3. Validación
        for (int t = 0; t < BATCH_SIZE; t++) {
            RiverState sCpu = resultCpu.getStateAt(t);
            RiverState sGpu = resultGpu.getStateAt(t);
            compareStates(t, sCpu, sGpu);
        }
        log.info(">> Paridad confirmada para {}", strategy);
    }

    private void compareStates(int step, RiverState cpu, RiverState gpu) {
        // Validamos la zona activa y un margen de la zona pasiva

        int checkLimit = Math.min(cellCount, BATCH_SIZE + 50);

        for (int i = 0; i < checkLimit; i++) {
            double hCpu = cpu.getWaterDepthAt(i);
            double hGpu = gpu.getWaterDepthAt(i);

            if (Math.abs(hCpu - hGpu) > EPSILON) {
                fail(String.format("Divergencia H en T=%d C=%d. CPU=%.5f GPU=%.5f", step, i, hCpu, hGpu));
            }

            double vCpu = cpu.getVelocityAt(i);
            double vGpu = gpu.getVelocityAt(i);

            if (Math.abs(vCpu - vGpu) > EPSILON) {
                fail(String.format("Divergencia V en T=%d C=%d. CPU=%.5f GPU=%.5f", step, i, vCpu, vGpu));
            }
        }
    }
}