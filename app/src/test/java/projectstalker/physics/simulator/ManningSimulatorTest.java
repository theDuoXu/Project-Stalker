package projectstalker.physics.simulator;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import projectstalker.config.RiverConfig;
import projectstalker.config.SimulationConfig;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;
import projectstalker.domain.simulation.ISimulationResult; // <--- Cambio a Interfaz

import java.lang.reflect.Field;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

/**
 * Test unitario para ManningSimulator.
 * <p>
 * REFACTORIZADO: Adaptado a la nueva arquitectura "Unified Batch Processor" e Interfaz ISimulationResult.
 * Verifica que el simulador orquesta correctamente el tiempo, genera los inputs (inputs 1D)
 * y delega la ejecución al ManningBatchProcessor.
 */
@Slf4j
class ManningSimulatorTest {

    private ManningSimulator simulator;
    private RiverConfig mockConfig;
    private SimulationConfig mockSimConfig;
    private RiverGeometry mockGeometry;
    private ManningBatchProcessor mockBatchProcessor;

    private final int CELL_COUNT = 5;
    private final double DELTA_TIME = 10.0;

    @BeforeEach
    void setUp() throws NoSuchFieldException, IllegalAccessException {
        // --- 1. Mocks Base ---
        mockConfig = mock(RiverConfig.class);
        when(mockConfig.totalLength()).thenReturn(25.0f);
        when(mockConfig.spatialResolution()).thenReturn(5.0f);
        when(mockConfig.seed()).thenReturn(12345L);
        when(mockConfig.baseManning()).thenReturn(0.035f);
        when(mockConfig.baseWidth()).thenReturn(10.0f);
        when(mockConfig.basePh()).thenReturn(7.0f);
        when(mockConfig.baseTemperature()).thenReturn(20.0f);
        when(mockConfig.noiseFrequency()).thenReturn(0.1f);

        mockSimConfig = mock(SimulationConfig.class);
        SimulationConfig.FlowConfig mockFlowConfig = mock(SimulationConfig.FlowConfig.class);
        when(mockFlowConfig.getBaseDischarge()).thenReturn(10.0f);
        when(mockSimConfig.getFlowConfig()).thenReturn(mockFlowConfig);
        when(mockSimConfig.isUseGpuAccelerationOnManning()).thenReturn(false);

        // --- 2. Mock de Geometría ---
        mockGeometry = mock(RiverGeometry.class);
        when(mockGeometry.getCellCount()).thenReturn(CELL_COUNT);
        when(mockGeometry.clonePhProfile()).thenReturn(new float[CELL_COUNT]);

        // --- 3. Instanciación e Inyección de Dependencias ---
        simulator = new ManningSimulator(mockConfig, mockSimConfig);

        Field geometryField = ManningSimulator.class.getDeclaredField("geometry");
        geometryField.setAccessible(true);
        geometryField.set(simulator, mockGeometry);

        mockBatchProcessor = mock(ManningBatchProcessor.class);
        Field batchProcessorField = ManningSimulator.class.getDeclaredField("batchProcessor");
        batchProcessorField.setAccessible(true);
        batchProcessorField.set(simulator, mockBatchProcessor);

        // --- 4. Estado Inicial ---
        simulator.setCurrentState(createDummyState(CELL_COUNT));
        simulator.setCurrentTimeInSeconds(100.0);
        log.info("ManningSimulator inicializado para pruebas.");
    }

    @AfterEach
    void tearDown() {
        simulator.close();
    }

    // --------------------------------------------------------------------------
    // TEST 1: Avance de Paso Único (Delegación a Batch=1)
    // --------------------------------------------------------------------------

    @Test
    @DisplayName("advanceTimeStep debe delegar a advanceBatchTimeStep(batchSize=1)")
    void advanceTimeStep_shouldDelegateToBatchProcessor() {
        log.info("Ejecutando test: advanceTimeStep (Single Step).");

        double initialTime = simulator.getCurrentTimeInSeconds();

        // Mock respuesta del processor (Interfaz Genérica)
        ISimulationResult mockResult = createMockResult(1);
        when(mockBatchProcessor.processBatch(eq(1), any(), any(), any(), anyBoolean()))
                .thenReturn(mockResult);

        // ACT
        simulator.advanceTimeStep(DELTA_TIME);

        // ASSERT
        verify(mockBatchProcessor, times(1)).processBatch(
                eq(1),
                any(RiverState.class),
                any(float[].class),
                any(float[][][].class),
                anyBoolean()
        );

        // Verifica avance de tiempo
        assertEquals(initialTime + DELTA_TIME, simulator.getCurrentTimeInSeconds(), 0.001);
    }

    // --------------------------------------------------------------------------
    // TEST 2: Avance por Lote (Batch)
    // --------------------------------------------------------------------------

    @Test
    @DisplayName("advanceBatchTimeStep debe generar inputs 1D y delegar al Processor")
    void advanceBatchTimeStep_shouldGenerate1DInputs_andDelegate() {
        log.info("Ejecutando test: advanceBatchTimeStep.");
        int batchSize = 3;
        double initialTime = simulator.getCurrentTimeInSeconds();

        // 1. Configurar Mock Resultado (Interfaz Genérica)
        ISimulationResult mockResult = createMockResult(batchSize);
        // Configuramos el mock para devolver un estado final dummy
        RiverState finalState = createDummyState(CELL_COUNT);
        // Simulamos una propiedad para verificar identidad después
        when(mockResult.getFinalState()).thenReturn(Optional.of(finalState));

        when(mockBatchProcessor.processBatch(eq(batchSize), any(), any(), any(), anyBoolean()))
                .thenReturn(mockResult);

        // Guardar el estado ANTES de ejecutar
        RiverState stateBeforeExecution = simulator.getCurrentState();

        // ACT
        ISimulationResult result = simulator.advanceBatchTimeStep(DELTA_TIME, batchSize);

        // ASSERT
        ArgumentCaptor<float[]> inflowCaptor = ArgumentCaptor.forClass(float[].class);

        // 1. Verificar Delegación
        verify(mockBatchProcessor, times(1)).processBatch(
                eq(batchSize),
                eq(stateBeforeExecution),
                inflowCaptor.capture(),
                any(float[][][].class),
                eq(false)
        );

        // 2. Verificar Inputs Generados
        float[] capturedInflows = inflowCaptor.getValue();
        assertEquals(batchSize, capturedInflows.length, "El array de inputs debe tener tamaño batchSize");
        assertEquals(10.0f, capturedInflows[0], 0.1f);

        // 3. Verificar Actualización de Estado Final
        // El simulador debe haber actualizado su currentState con lo que devolvió el mock
        assertEquals(finalState, simulator.getCurrentState());
        assertEquals(initialTime + (batchSize * DELTA_TIME), simulator.getCurrentTimeInSeconds(), 0.001);
    }

    // --- Helpers ---

    private RiverState createDummyState(int n) {
        float[] data = new float[n];
        return new RiverState(data, data, data, data, data);
    }

    // Helper simplificado: Mockeamos la interfaz directamente
    private ISimulationResult createMockResult(int batchSize) {
        ISimulationResult result = mock(ISimulationResult.class);
        when(result.getTimestepCount()).thenReturn(batchSize);
        // Configuración básica para evitar NPE si el código bajo prueba llama a getFinalState
        when(result.getFinalState()).thenReturn(Optional.of(createDummyState(CELL_COUNT)));
        return result;
    }
}