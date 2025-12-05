package projectstalker.physics.simulator;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import projectstalker.config.RiverConfig;
import projectstalker.config.SimulationConfig;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;
import projectstalker.domain.simulation.ISimulationResult;
import projectstalker.factory.RiverGeometryFactory;
import projectstalker.physics.model.RiverPhModel;
import projectstalker.physics.model.RiverTemperatureModel;
import projectstalker.config.SimulationConfig.GpuStrategy;

import java.util.Arrays;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

/**
 * Test unitario para ManningBatchProcessor.
 * <p>
 * Adaptado a la nueva arquitectura Stateful/Smart Fetch e Interfaz ISimulationResult.
 * Verifica la lógica de ensamblaje en modo CPU (Legacy/Integrity Check) utilizando
 * modelos fisicoquímicos reales.
 */
@Slf4j
class ManningBatchProcessorTest {

    private ManningBatchProcessor batchProcessor;
    private RiverGeometry realGeometry;
    private RiverTemperatureModel realTempModel;
    private RiverPhModel realPhModel;
    private SimulationConfig mockConfig;

    private int cellCount;
    private final int BATCH_SIZE = 3;
    private final double DELTA_TIME = 10.0;

    @BeforeEach
    void setUp() {
        // --- 1. Inicializar INSTANCIA REAL de Geometría y Configuración ---
        RiverConfig config = RiverConfig.getTestingRiver();
        RiverGeometryFactory factory = new RiverGeometryFactory();
        this.realGeometry = factory.createRealisticRiver(config);

        // --- 2. Inicializar INSTANCIAS REALES de Modelos Fisicoquímicos ---
        this.realTempModel = new RiverTemperatureModel(config, this.realGeometry);
        this.realPhModel = new RiverPhModel(this.realGeometry);
        this.cellCount = this.realGeometry.getCellCount();

        // --- 3. Inicializar Mocks de Configuración ---
        mockConfig = mock(SimulationConfig.class);
        when(mockConfig.getCpuProcessorCount()).thenReturn(2);
        when(mockConfig.isUseGpuAccelerationOnManning()).thenReturn(false);

        // --- 4. Inicializar BatchProcessor ---
        batchProcessor = new ManningBatchProcessor(this.realGeometry, mockConfig);
        log.info("ManningBatchProcessor inicializado.");
    }

    @AfterEach
    void tearDown() {
        if (batchProcessor != null) {
            batchProcessor.close();
        }
    }

    @Test
    @DisplayName("El batch en modo CPU debe ejecutar tareas y ensamblar resultados coherentes (API 1D)")
    void processBatch_cpuMode_shouldExecuteAndAssembleResults() {
        log.info("Iniciando test: processBatch en modo CPU concurrente. BATCH_SIZE={}", BATCH_SIZE);

        // --- ARRANGE ---
        double currentTime = 300.0 * 3600.0;

        // 1. Estado Inicial
        float initialUniformDepth = 0.5f;
        float[] initialData = new float[cellCount];
        Arrays.fill(initialData, initialUniformDepth);
        float[] initialVel = new float[cellCount];
        Arrays.fill(initialVel, 1.0f);

        RiverState initialRiverState = new RiverState(
                initialData, initialVel, initialData, initialData, initialData
        );

        // 2. Inputs de Caudal
        float[] newInflows = new float[BATCH_SIZE];
        Arrays.fill(newInflows, 200.0f);

        // 3. Resultados Fisicoquímicos
        float[][][] phTmp = new float[BATCH_SIZE][2][cellCount];
        float timeStep = 0.0f;
        for (int i = 0; i < BATCH_SIZE; i++) {
            double t = currentTime + timeStep;
            phTmp[i][0] = realTempModel.calculate(t);
            phTmp[i][1] = realPhModel.getPhProfile();
            timeStep += DELTA_TIME;
        }

        // --- ACT ---
        // Actualizado para pasar la Estrategia (aunque en CPU se ignora)
        ISimulationResult result = batchProcessor.processBatch(
                BATCH_SIZE,
                initialRiverState,
                newInflows,
                phTmp,
                false, // isGpuAccelerated = FALSE (Modo CPU)
                GpuStrategy.SMART_SAFE // Estrategia (Irrelevante aquí, pero requerida por firma)
        );

        // --- ASSERT ---
        assertNotNull(result, "El resultado no debe ser nulo.");
        assertEquals(BATCH_SIZE, result.getTimestepCount());

        RiverState finalState = result.getStateAt(BATCH_SIZE - 1);
        double finalAvgDepth = calculateAverageDepth(finalState);
        double finalAvgVelocity = calculateAverageVelocity(finalState);

        log.info("Estadísticas Finales CPU -> H_avg: {} m | V_avg: {} m/s", finalAvgDepth, finalAvgVelocity);

        double initialAvgDepth = calculateAverageDepth(initialRiverState);
        assertTrue(finalAvgDepth > initialAvgDepth, "La profundidad debería aumentar con el influjo.");

        RiverState state0 = result.getStateAt(0);
        double expectedTemp0 = phTmp[0][0][0];
        assertEquals(expectedTemp0, state0.getTemperatureAt(0), 1e-6);
    }

    private double calculateAverageDepth(RiverState state) {
        if (state.waterDepth().length == 0) return 0.0;
        return IntStream.range(0, state.waterDepth().length).mapToDouble(i->state.waterDepth()[i]).average().orElse(0.0);
    }

    private double calculateAverageVelocity(RiverState state) {
        if (state.velocity().length == 0) return 0.0;
        return IntStream.range(0, state.velocity().length).mapToDouble(i->state.velocity()[i]).average().orElse(0.0);
    }
}