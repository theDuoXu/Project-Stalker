package projectstalker.physics.simulator;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import projectstalker.config.RiverConfig;
import projectstalker.config.SimulationConfig;
import projectstalker.config.SimulationConfig.GpuStrategy;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;
import projectstalker.domain.simulation.ISimulationResult;
import projectstalker.factory.RiverGeometryFactory;
import projectstalker.physics.model.RiverPhModel;
import projectstalker.physics.model.RiverTemperatureModel;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test de Integración End-to-End para el solver de GPU.
 * Verifica la carga de librería nativa y la ejecución del ciclo completo (Init->Run->Destroy).
 */
@Tag("GPU")
@Slf4j
class ManningGpuIntegrationTest {

    private ManningBatchProcessor batchProcessor;
    private RiverGeometry realGeometry;
    private RiverTemperatureModel realTempModel;
    private RiverPhModel realPhModel;
    private SimulationConfig simConfig;

    private int cellCount;
    private final int BATCH_SIZE = 3;
    private final double DELTA_TIME = 10.0;

    @BeforeEach
    void setUp() {
        log.info("Configurando entorno para Test de Integración GPU...");

        RiverConfig riverConfig = RiverConfig.getTestingRiver();
        RiverGeometryFactory factory = new RiverGeometryFactory();
        this.realGeometry = factory.createRealisticRiver(riverConfig);
        this.cellCount = this.realGeometry.getCellCount();

        this.realTempModel = new RiverTemperatureModel(riverConfig, this.realGeometry);
        this.realPhModel = new RiverPhModel(this.realGeometry);

        this.simConfig = SimulationConfig.builder()
                .riverConfig(riverConfig)
                .seed(12345L)
                .totalTime(3600)
                .deltaTime((float) DELTA_TIME)
                .cpuProcessorCount(2)
                .cpuTimeBatchSize(BATCH_SIZE)
                .useGpuAccelerationOnManning(true) // Activar GPU
                .useGpuAccelerationOnTransport(false)
                .build();

        log.info("Entorno configurado. Geometría con {} celdas.", this.realGeometry.getCellCount());
    }

    @AfterEach
    void tearDown() {
        if (batchProcessor != null) {
            batchProcessor.close();
        }
    }

    @Test
    @DisplayName("Integration SMART: El modo optimizado debe ejecutar sin crashear")
    void integrationTest_SmartMode() {
        runIntegrationTest(GpuStrategy.SMART_TRUSTED);
    }

    @Test
    @DisplayName("Integration FULL: El modo robusto debe ejecutar sin crashear")
    void integrationTest_FullMode() {
        runIntegrationTest(GpuStrategy.FULL_EVOLUTION);
    }

    private void runIntegrationTest(GpuStrategy strategy) {
        log.info(">>> INICIANDO INTEGRATION TEST: {} <<<", strategy);

        // 1. Estado Inicial Dummy
        float[] initialDepth = new float[this.cellCount]; Arrays.fill(initialDepth, 0.5f);
        float[] initialVel = new float[this.cellCount]; Arrays.fill(initialVel, 1.0f);
        float[] zeros = new float[this.cellCount];

        RiverState initialRiverState = new RiverState(
                initialDepth, initialVel, zeros, zeros, zeros
        );

        // 2. Inputs
        float[] newDischarges = new float[BATCH_SIZE];
        Arrays.fill(newDischarges, 200.0f);

        // 3. Instanciar SUT
        batchProcessor = new ManningBatchProcessor(this.realGeometry, simConfig);

        // 4. Auxiliares
        float[][][] phTmp = new float[BATCH_SIZE][2][this.cellCount];
        for (int i = 0; i < BATCH_SIZE; i++) {
            phTmp[i][0] = realTempModel.calculate(0);
            phTmp[i][1] = realPhModel.getPhProfile();
        }

        // --- ACT ---
        ISimulationResult result = assertDoesNotThrow(() ->
                        batchProcessor.processBatch(
                                BATCH_SIZE,
                                initialRiverState,
                                newDischarges,
                                phTmp,
                                true, // GPU
                                strategy // Estrategia explícita
                        ),
                "Fallo crítico en ejecución nativa JNI/CUDA"
        );

        // --- ASSERT ---
        assertNotNull(result);

        // Verificación de integridad básica
        // En modo Full con Stride por defecto (1), tendremos todos los pasos.
        // En modo Smart, también.
        assertEquals(BATCH_SIZE, result.getTimestepCount());

        RiverState finalState = result.getStateAt(BATCH_SIZE - 1);

        // Chequeo de sanidad numérica (No NaN)
        float h = finalState.getWaterDepthAt(0);
        assertFalse(Float.isNaN(h), "El resultado contiene NaNs");
        assertTrue(h > 0, "La profundidad debe ser positiva");

        log.info(">>> TEST {} SUPERADO <<<", strategy);
    }
}