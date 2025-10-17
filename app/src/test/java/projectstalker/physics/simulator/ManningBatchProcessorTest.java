package projectstalker.physics.simulator;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
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
import static org.mockito.Mockito.*;

/**
 * Test unitario para ManningBatchProcessor, utilizando instancias reales de los
 * modelos fisicoquímicos (RiverTemperatureModel y RiverPhModel) y de RiverGeometry.
 * Esto garantiza que la lógica de ensamblaje (que combina resultados hidrológicos con
 * resultados fisicoquímicos) es probada con datos realistas generados por los modelos.
 */
@Slf4j
class ManningBatchProcessorTest {

    private ManningBatchProcessor batchProcessor;
    private RiverGeometry realGeometry;
    private RiverTemperatureModel realTempModel; // INSTANCIA REAL
    private RiverPhModel realPhModel;           // INSTANCIA REAL
    private SimulationConfig mockConfig;

    // Dimensiones predichas para el río (100000m / 50m = 2000 celdas)
    private final int CELL_COUNT = 2000;
    private final int BATCH_SIZE = 3;
    private final double DELTA_TIME = 10.0; // Intervalo de tiempo para el cálculo del batch

    @BeforeEach
    void setUp() {
        // --- 1. Inicializar INSTANCIA REAL de Geometría y Configuración ---
        RiverConfig config = new RiverConfig(
                12345L, 0.0f, 0.05f, 0.001f, 100000.0, 50.0, 200.0, 0.4, 0.0002,
                0.0001, 150.0, 40.0, 4.0, 1.5, 0.030, 0.005, 0.1, 0.05,
                15.0, 2.0, 8.0, 14.0, 7.5, 0.5,
                4.0, 20000.0, 1.5, 1.0, 0.25
        );
        RiverGeometryFactory factory = new RiverGeometryFactory();
        this.realGeometry = factory.createRealisticRiver(config);

        // --- 2. Inicializar INSTANCIAS REALES de Modelos Fisicoquímicos ---
        this.realTempModel = new RiverTemperatureModel(config, this.realGeometry);
        this.realPhModel = new RiverPhModel(this.realGeometry);

        // --- 3. Inicializar Mocks de Configuración ---
        mockConfig = mock(SimulationConfig.class);
        when(mockConfig.getCpuProcessorCount()).thenReturn(2);
        when(mockConfig.isUseGpuAccelerationOnManning()).thenReturn(false);

        // --- 4. Inicializar BatchProcessor con la Geometría y Modelos REALES ---
        batchProcessor = new ManningBatchProcessor(this.realGeometry, mockConfig);
        log.info("ManningBatchProcessor inicializado con instancias reales de Geometry, TempModel, y PhModel.");
    }

    @Test
    @DisplayName("El batch en modo CPU debe ejecutar tareas y ensamblar resultados coherentes, usando modelos reales")
    void processBatch_cpuMode_shouldExecuteAndAssembleResults() throws Exception {
        log.info("Iniciando test: processBatch en modo CPU concurrente. BATCH_SIZE={}", BATCH_SIZE);

        // --- ARRANGE: Preparación de Datos ---
        double currentTime = 300.0 * 3600.0;

        // 1. Estado Inicial
        double initialUniformDepth = 0.5;
        double[] initialData = new double[CELL_COUNT];
        Arrays.fill(initialData, initialUniformDepth);

        RiverState initialRiverState = new RiverState(
                initialData, initialData, initialData, initialData
        );

        // Logging del estado ANTES de la simulación
        double initialAvgDepth = calculateAverageDepth(initialRiverState);
        double initialAvgVelocity = calculateAverageVelocity(initialRiverState);
        log.info("======================== ESTADO INICIAL ========================");
        log.info("Caudal de Entrada al Batch (Q_in): {} m³/s", 200.0);
        log.info("Tamaño de batch: {} m³/s", BATCH_SIZE);
        log.info("Profundidad Media Inicial (H_avg): {} m", initialAvgDepth);
        log.info("Velocidad Media Inicial (V_avg): {} m/s", initialAvgVelocity);
        log.info("==============================================================");

        // 2. Perfiles de Caudal
        double[] newDischarges = new double[BATCH_SIZE];
        Arrays.fill(newDischarges, 200.0);
        double[] initialDischarges = new double[CELL_COUNT];

        Arrays.fill(initialDischarges, this.realGeometry.getCrossSectionalArea(0, 0.5) * 0.5); // initialUniformDepth = 0.5; y lo hemos metido en todo, por lo que v=0.5 y profundidad = 0.5
        double[][] allDischargeProfiles = batchProcessor.createDischargeProfiles(BATCH_SIZE, newDischarges, initialDischarges);

        // 3. Resultados Fisicoquímicos (Pre-cálculo)
        double[][][] phTmp = new double[BATCH_SIZE][2][CELL_COUNT];
        double timeStep = 0.0;
        for (int i = 0; i < BATCH_SIZE; i++) {
            double t = currentTime + timeStep;
            phTmp[i][0] = realTempModel.calculate(t);
            phTmp[i][1] = realPhModel.getPhProfile();
            timeStep += DELTA_TIME;
        }

        // --- ACT: Ejecución Real del Batch Processor ---
        ManningSimulationResult result = batchProcessor.processBatch(
                BATCH_SIZE, currentTime, initialRiverState,
                allDischargeProfiles, phTmp, false
        );

        // --- ASSERT Y LOGGING DE RESULTADOS ---
        assertNotNull(result, "El resultado de la simulación no debe ser nulo.");
        assertEquals(BATCH_SIZE, result.getStates().size(), "El resultado debe tener el tamaño de batch correcto.");

        // Obtener y loggear el estado DESPUÉS del batch (el último estado)
        RiverState finalState = result.getStates().get(BATCH_SIZE - 1);
        double finalAvgDepth = calculateAverageDepth(finalState);
        double finalAvgVelocity = calculateAverageVelocity(finalState);

        log.info("======================== ESTADO FINAL (Paso {}) ========================", BATCH_SIZE - 1);
        log.info("Profundidad Media Final (H_avg): {} m", finalAvgDepth);
        log.info("Velocidad Media Final (V_avg): {} m/s", finalAvgVelocity);
        log.info("===================================================================");

        // 1. Verificación de Lógica Hidrológica (Aumento de profundidad y velocidad = aumento de volumen)
        assertTrue(finalAvgDepth*finalAvgVelocity > initialAvgDepth*initialAvgVelocity, "El caudal medio debe haber aumentado");

        // 2. Verificación de Ensamblaje Fisicoquímico
        RiverState state0 = result.getStates().get(0);
        double expectedTemp0 = phTmp[0][0][0];
        assertEquals(expectedTemp0, state0.getTemperatureAt(0), 1e-6, "La temperatura ensamblada debe coincidir con la calculada por el modelo real.");

        log.info("Cálculo completo (hidrología + fisicoquímica real) verificado.");
    }

    /**
     * Calcula la profundidad media del río a partir del estado de un RiverState.
     */
    private double calculateAverageDepth(RiverState state) {
        if (state.waterDepth().length == 0) return 0.0;
        return Arrays.stream(state.waterDepth()).average().orElse(0.0);
    }

    /**
     * Calcula la velocidad media del río a partir del estado de un RiverState.
     */
    private double calculateAverageVelocity(RiverState state) {
        if (state.velocity().length == 0) return 0.0;
        return Arrays.stream(state.velocity()).average().orElse(0.0);
    }
}