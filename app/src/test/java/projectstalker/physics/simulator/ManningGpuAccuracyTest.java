package projectstalker.physics.simulator;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import projectstalker.config.RiverConfig;
import projectstalker.config.SimulationConfig;
import projectstalker.config.SimulationConfig.GpuStrategy;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;
import projectstalker.domain.simulation.IManningResult;
import projectstalker.factory.RiverFactory;
import projectstalker.factory.RiverGeometryFactory;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test de Integración para validar la precisión numérica de la GPU frente a la CPU.
 * <p>
 * REFACTORIZADO: Adaptado a la arquitectura de Orquestador de Lotes (BatchProcessor v2).
 * Ejecuta una simulación corta y compara celda a celda los resultados.
 */
@Tag("GPU")
@Slf4j
class ManningGpuAccuracyTest {

    private RiverGeometry realGeometry;
    private SimulationConfig cpuConfig;
    private SimulationConfig baseGpuConfig;

    private int cellCount;
    private final int BATCH_SIZE = 5;

    // Tolerancia (Float GPU vs Double/Float CPU)
    // Manning es sensible, 1cm de error es aceptable en simulaciones rápidas.
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

        // 1. Generar Estado Estable (Usando la nueva Factory Analítica)
        this.stableInitialState = RiverFactory.createSteadyState(realGeometry, BASE_DISCHARGE);

        // 2. Configuración Base
        SimulationConfig baseConfig = SimulationConfig.builder()
                .riverConfig(riverConfig)
                .seed(12345L)
                .totalTime(3600)
                .deltaTime(10.0f)
                .cpuProcessorCount(4)
                .cpuTimeBatchSize(BATCH_SIZE) // Importante para que el procesador sepa cortar
                .useGpuAccelerationOnTransport(false)
                .build();

        // Configuración CPU
        this.cpuConfig = baseConfig.withUseGpuAccelerationOnManning(false);

        // Configuración GPU Base (Full Evolution Stride 1 para comparar paso a paso)
        this.baseGpuConfig = baseConfig
                .withUseGpuAccelerationOnManning(true)
                .withGpuFullEvolutionStride(1);
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

        // Input: Onda de Avenida constante (Step Input)
        float[] flowInput = new float[BATCH_SIZE];
        Arrays.fill(flowInput, 150.0f);

        IManningResult resultCpu;
        IManningResult resultGpu;

        // 1. CPU Reference
        // Instanciamos un procesador configurado para CPU
        try (ManningBatchProcessor cpuProcessor = new ManningBatchProcessor(realGeometry, cpuConfig)) {
            resultCpu = cpuProcessor.process(flowInput, initialState);
        }

        // 2. GPU SUT (System Under Test)
        // Instanciamos un procesador configurado con la estrategia específica
        SimulationConfig specificGpuConfig = baseGpuConfig.withGpuStrategy(strategy);

        try (ManningBatchProcessor gpuProcessor = new ManningBatchProcessor(realGeometry, specificGpuConfig)) {
            log.info("Ejecutando GPU con estrategia: {}", strategy);
            resultGpu = gpuProcessor.process(flowInput, initialState);
        }

        // 3. Validación
        // Comprobamos que ambos resultados tengan el mismo número de pasos
        assertEquals(BATCH_SIZE, resultCpu.getTimestepCount(), "Pasos CPU incorrectos");
        assertEquals(BATCH_SIZE, resultGpu.getTimestepCount(), "Pasos GPU incorrectos");

        for (int t = 0; t < BATCH_SIZE; t++) {
            // Aquí la magia del polimorfismo de IManningResult actúa:
            // resultCpu será DenseManningResult
            // resultGpu será FlyweightManningResult (Smart) o StridedManningResult (Full)
            RiverState sCpu = resultCpu.getStateAt(t);
            RiverState sGpu = resultGpu.getStateAt(t);

            compareStates(t, sCpu, sGpu);
        }
        log.info(">> Paridad confirmada para {}", strategy);
    }

    private void compareStates(int step, RiverState cpu, RiverState gpu) {
        // Validamos la zona activa y un margen de la zona pasiva
        // En Smart, solo se calcula hasta donde llega la ola, el resto es copia del inicial.
        // En Full, se calcula todo.
        // Ambos deberían coincidir con la CPU.

        int checkLimit = cellCount;

        for (int i = 0; i < checkLimit; i++) {
            float hCpu = cpu.getWaterDepthAt(i);
            float hGpu = gpu.getWaterDepthAt(i);

            // Verificación de NaN (Fallo catastrófico GPU)
            if (Float.isNaN(hGpu)) {
                fail(String.format("GPU produjo NaN en T=%d C=%d", step, i));
            }

            if (Math.abs(hCpu - hGpu) > EPSILON) {
                fail(String.format("Divergencia H en T=%d C=%d. CPU=%.5f GPU=%.5f Delta=%.5f",
                        step, i, hCpu, hGpu, Math.abs(hCpu - hGpu)));
            }

            float vCpu = cpu.getVelocityAt(i);
            float vGpu = gpu.getVelocityAt(i);

            if (Math.abs(vCpu - vGpu) > EPSILON) {
                fail(String.format("Divergencia V en T=%d C=%d. CPU=%.5f GPU=%.5f Delta=%.5f",
                        step, i, vCpu, vGpu, Math.abs(vCpu - vGpu)));
            }
        }
    }
}