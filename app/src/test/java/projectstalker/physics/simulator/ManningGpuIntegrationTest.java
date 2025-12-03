package projectstalker.physics.simulator;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.AfterEach; // Nuevo
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import projectstalker.config.RiverConfig;
import projectstalker.config.SimulationConfig;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;
import projectstalker.domain.simulation.ManningSimulationResult;
import projectstalker.factory.RiverGeometryFactory;
import projectstalker.physics.model.RiverPhModel;
import projectstalker.physics.model.RiverTemperatureModel;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.mock;

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
        // Crucial: Liberar la GPU al terminar el test
        if (batchProcessor != null) {
            batchProcessor.close();
        }
    }

    @Test
    @DisplayName("El batch en modo GPU debe cargar la librería nativa y ejecutar sin crashear")
    void processBatch_gpuMode_shouldRunOnNativeLibraryWithoutCrashing() {
        log.info("Iniciando test de integración de GPU. BATCH_SIZE={}", BATCH_SIZE);

        double currentTime = 300.0 * 3600.0;

        // 1. Estado Inicial
        float[] initialDepth = new float[this.cellCount]; Arrays.fill(initialDepth, 0.5f);
        float[] initialVel = new float[this.cellCount]; Arrays.fill(initialVel, 1.0f);
        float[] zeros = new float[this.cellCount];

        RiverState initialRiverState = new RiverState(
                initialDepth, initialVel, zeros, zeros, zeros
        );

        // 2. Inputs (1D - Smart Fetch)
        float[] newDischarges = new float[BATCH_SIZE];
        Arrays.fill(newDischarges, 200.0f); // Caudal entrante

        // 3. Instanciar el SUT (Carga nativa + Init Session)
        log.info("Instanciando ManningBatchProcessor (Carga librería + Init GPU)...");
        batchProcessor = assertDoesNotThrow(
                () -> new ManningBatchProcessor(this.realGeometry, simConfig),
                "Falló la instanciación. ¿Librería nativa faltante o error en initSession?"
        );

        // 4. Resultados Fisicoquímicos (Pre-cálculo)
        float[][][] phTmp = new float[BATCH_SIZE][2][this.cellCount];
        double timeStep = 0.0;
        for (int i = 0; i < BATCH_SIZE; i++) {
            double t = currentTime + timeStep;
            phTmp[i][0] = realTempModel.calculate(t);
            phTmp[i][1] = realPhModel.getPhProfile();
            timeStep += DELTA_TIME;
        }

        // --- ACT & ASSERT (Ejecución Nativa) ---
        log.warn(">>> EJECUTANDO STACK NATIVO (GPU) <<<");

        ManningSimulationResult result = assertDoesNotThrow(() ->
                        batchProcessor.processBatch(
                                BATCH_SIZE,
                                initialRiverState,
                                newDischarges, // Input 1D
                                phTmp,
                                true // GPU Habilitada
                        ),
                "La ejecución nativa de la GPU falló (Crashes o Excepciones JNI/CUDA)"
        );

        log.warn(">>> STACK NATIVO EJECUTADO EXITOSAMENTE <<<");

        // --- ASSERT (Sanity Checks) ---
        assertNotNull(result, "Resultado nulo.");
        assertEquals(BATCH_SIZE, result.getStates().size());

        RiverState finalState = result.getStates().get(BATCH_SIZE - 1);

        // Verificación de integridad numérica
        float sampleH = finalState.getWaterDepthAt(10);
        float sampleV = finalState.getVelocityAt(10);

        assertFalse(Float.isNaN(sampleH), "Profundidad NaN detectada.");
        assertFalse(Float.isNaN(sampleV), "Velocidad NaN detectada.");
        assertTrue(sampleH > 0.0f, "Profundidad debe ser positiva.");
        assertTrue(sampleV > 0.0f, "Velocidad debe ser positiva.");

        log.info("Test de Integración GPU superado.");
    }
}