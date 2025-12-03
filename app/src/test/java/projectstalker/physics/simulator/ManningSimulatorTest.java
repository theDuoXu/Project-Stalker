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
import projectstalker.domain.simulation.ManningSimulationResult;

import java.lang.reflect.Field;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

/**
 * Test unitario para ManningSimulator.
 * <p>
 * REFACTORIZADO: Adaptado a la nueva arquitectura "Unified Batch Processor".
 * Verifica que el simulador orquesta correctamente el tiempo, genera los inputs (inputs 1D)
 * y delega la ejecución al ManningBatchProcessor.
 */
@Slf4j
class ManningSimulatorTest {

    private ManningSimulator simulator;
    private RiverConfig mockConfig;
    private SimulationConfig mockSimConfig;
    private RiverGeometry mockGeometry;
    private ManningBatchProcessor mockBatchProcessor; // El ÚNICO delegado ahora

    private final int CELL_COUNT = 5;
    private final double DELTA_TIME = 10.0;

    @BeforeEach
    void setUp() throws NoSuchFieldException, IllegalAccessException {
        // --- 1. Mocks Base ---
        mockConfig = mock(RiverConfig.class);
        when(mockConfig.totalLength()).thenReturn(25.0f);
        when(mockConfig.spatialResolution()).thenReturn(5.0f);
        // Parámetros dummy para evitar NPEs en factorías
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
        // Creamos el simulador real
        simulator = new ManningSimulator(mockConfig, mockSimConfig);

        // Inyectamos el Mock de Geometría (necesario para validaciones de tamaño)
        Field geometryField = ManningSimulator.class.getDeclaredField("geometry");
        geometryField.setAccessible(true);
        geometryField.set(simulator, mockGeometry);

        // Inyectamos el Mock de BatchProcessor (El corazón de la refactorización)
        // Ahora el simulador NO tiene gpuSolver ni cpuSolver directos, todo va al processor.
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

        // Mock respuesta del processor
        ManningSimulationResult mockResult = createMockResult(1);
        when(mockBatchProcessor.processBatch(eq(1), any(), any(), any(), anyBoolean()))
                .thenReturn(mockResult);

        // ACT
        simulator.advanceTimeStep(DELTA_TIME);

        // ASSERT
        // Verifica que se llamó a processBatch con size=1 y un array de inputs de tamaño 1
        verify(mockBatchProcessor, times(1)).processBatch(
                eq(1),
                any(RiverState.class),
                any(float[].class), // newInflows (1D)
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

        // 1. Configurar Mock Resultado
        ManningSimulationResult mockResult = createMockResult(batchSize);
        // Marcamos el último estado con un valor reconocible
        RiverState finalState = mockResult.getStates().get(batchSize - 1);
        when(finalState.getWaterDepthAt(0)).thenReturn(99.9f); // Marker

        when(mockBatchProcessor.processBatch(eq(batchSize), any(), any(), any(), anyBoolean()))
                .thenReturn(mockResult);

        // Guardar el estado ANTES de ejecutar ---
        RiverState stateBeforeExecution = simulator.getCurrentState();

        // ACT
        ManningSimulationResult result = simulator.advanceBatchTimeStep(DELTA_TIME, batchSize);

        // ASSERT
        ArgumentCaptor<float[]> inflowCaptor = ArgumentCaptor.forClass(float[].class);

        // 1. Verificar Delegación con nueva firma
        verify(mockBatchProcessor, times(1)).processBatch(
                eq(batchSize),
                eq(stateBeforeExecution),        // <-- USAR LA REFERENCIA GUARDADA (Inicial)
                inflowCaptor.capture(),          // Inputs 1D capturados
                any(float[][][].class),          // phTmp
                eq(false)                        // isGpuAccelerated
        );

        // 2. Verificar Inputs Generados (Smart Fetch Array)
        float[] capturedInflows = inflowCaptor.getValue();
        assertEquals(batchSize, capturedInflows.length, "El array de inputs debe tener tamaño batchSize");
        // El FlowConfig mock devuelve 10.0f base, así que deberíamos ver eso
        assertEquals(10.0f, capturedInflows[0], 0.1f);

        // 3. Verificar Actualización de Estado Final
        // Aquí SÍ queremos verificar que el simulador tiene el estado NUEVO
        assertEquals(finalState, simulator.getCurrentState());
        assertEquals(initialTime + (batchSize * DELTA_TIME), simulator.getCurrentTimeInSeconds(), 0.001);
    }

    // --- Helpers ---

    private RiverState createDummyState(int n) {
        float[] data = new float[n];
        return new RiverState(data, data, data, data, data);
    }

    private ManningSimulationResult createMockResult(int batchSize) {
        // Crear lista de estados mockeados
        List<RiverState> states = new java.util.ArrayList<>();
        for(int i=0; i<batchSize; i++) {
            states.add(mock(RiverState.class));
        }

        return ManningSimulationResult.builder()
                .geometry(mockGeometry)
                .states(states)
                .build();
    }
}