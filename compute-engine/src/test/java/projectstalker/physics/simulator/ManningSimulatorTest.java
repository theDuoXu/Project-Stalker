package projectstalker.physics.simulator;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import projectstalker.config.RiverConfig;
import projectstalker.config.SimulationConfig;
import projectstalker.config.SimulationConfig.GpuStrategy;
import projectstalker.domain.river.RiverState;
import projectstalker.domain.simulation.IManningResult;

import java.lang.reflect.Field;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

/**
 * Test unitario para ManningSimulator.
 * <p>
 * REFACTORIZADO: Adaptado a la nueva arquitectura de "Orquestación Completa".
 * Verifica que el simulador genera el hidrograma completo y lo delega al BatchProcessor.
 */
@Slf4j
class ManningSimulatorTest {

    private ManningSimulator simulator;
    private RiverConfig mockRiverConfig;
    private SimulationConfig mockSimConfig;
    private ManningBatchProcessor mockBatchProcessor;

    private final float DELTA_TIME = 10.0f;
    private final long TOTAL_TIME = 1000L; // 100 pasos
    private final int EXPECTED_STEPS = 100;

    @BeforeEach
    void setUp() throws Exception {
        // --- 1. Mocks de Configuración ---
        mockRiverConfig = mock(RiverConfig.class);
        // Valores necesarios para que RiverGeometryFactory (interna) no falle
        when(mockRiverConfig.totalLength()).thenReturn(100.0f);
        when(mockRiverConfig.spatialResolution()).thenReturn(10.0f); // 10 celdas
        when(mockRiverConfig.seed()).thenReturn(12345L);
        when(mockRiverConfig.baseManning()).thenReturn(0.035f);
        when(mockRiverConfig.baseWidth()).thenReturn(10.0f);
        when(mockRiverConfig.averageSlope()).thenReturn(0.001f);
        when(mockRiverConfig.baseSideSlope()).thenReturn(1.0f);

        // Mocks para modelos químicos (aunque no se usen en simulación pura, se instancian)
        when(mockRiverConfig.basePh()).thenReturn(7.0f);
        when(mockRiverConfig.baseTemperature()).thenReturn(20.0f);
        when(mockRiverConfig.noiseFrequency()).thenReturn(0.1f);

        // Simulation Config
        mockSimConfig = mock(SimulationConfig.class);
        when(mockSimConfig.getTotalTimeSteps()).thenReturn(TOTAL_TIME / (long)DELTA_TIME);
        when(mockSimConfig.getDeltaTime()).thenReturn(DELTA_TIME);

        // Flow Config
        SimulationConfig.FlowConfig mockFlowConfig = mock(SimulationConfig.FlowConfig.class);
        when(mockFlowConfig.getBaseDischarge()).thenReturn(10.0f);
        when(mockSimConfig.getFlowConfig()).thenReturn(mockFlowConfig);

        // Config GPU
        when(mockSimConfig.isUseGpuAccelerationOnManning()).thenReturn(true);
        when(mockSimConfig.getGpuStrategy()).thenReturn(GpuStrategy.FULL_EVOLUTION);
        when(mockSimConfig.getCpuProcessorCount()).thenReturn(1);

        // --- 2. Instanciación ---
        // Al hacer new, el simulador crea internamente un BatchProcessor real y un RiverGeometry real.
        // También llama a RiverFactory.createSteadyState (que es estático y rápido, así que lo dejamos correr).
        simulator = new ManningSimulator(mockRiverConfig, mockSimConfig);

        // --- 3. Inyección de Mock BatchProcessor ---
        // Necesitamos interceptar la llamada al process() para verificar la orquestación.
        mockBatchProcessor = mock(ManningBatchProcessor.class);

        Field batchProcessorField = ManningSimulator.class.getDeclaredField("batchProcessor");
        batchProcessorField.setAccessible(true);
        // Reemplazamos el procesador real (que se creó en el constructor) por el mock
        batchProcessorField.set(simulator, mockBatchProcessor);

        log.info("ManningSimulator inicializado con Mock Processor inyectado.");
    }

    @AfterEach
    void tearDown() {
        if (simulator != null) simulator.close();
    }

    // --------------------------------------------------------------------------
    // TEST: Ejecución Completa (RunFullSimulation)
    // --------------------------------------------------------------------------

    @Test
    @DisplayName("runFullSimulation: Debe generar hidrograma completo y delegar al processor")
    void runFullSimulation_ShouldGenerateHydrographAndDelegate() {
        // ARRANGE
        // Configuramos el mock processor para devolver un resultado dummy
        IManningResult mockResult = mock(IManningResult.class);
        when(mockResult.getSimulationTime()).thenReturn(50L);

        when(mockBatchProcessor.process(any(float[].class), any(RiverState.class)))
                .thenReturn(mockResult);

        // ACT
        IManningResult result = simulator.runFullSimulation();

        // ASSERT
        assertNotNull(result);

        // 1. Capturamos los argumentos pasados al procesador
        ArgumentCaptor<float[]> inflowCaptor = ArgumentCaptor.forClass(float[].class);
        ArgumentCaptor<RiverState> stateCaptor = ArgumentCaptor.forClass(RiverState.class);

        verify(mockBatchProcessor, times(1)).process(inflowCaptor.capture(), stateCaptor.capture());

        // 2. Verificamos el hidrograma generado
        float[] generatedInflows = inflowCaptor.getValue();
        assertEquals(EXPECTED_STEPS, generatedInflows.length,
                "El simulador debe generar un array de inputs para todos los pasos de tiempo.");

        // Verificamos que el generador de flujo se usó (no es todo cero)
        // Como baseDischarge es 10.0, deberíamos tener valores cercanos a 10
        assertEquals(10.0f, generatedInflows[0], 2.0f);

        // 3. Verificamos el estado inicial
        RiverState passedState = stateCaptor.getValue();
        assertNotNull(passedState, "Debe pasar un estado inicial válido");
        // El estado inicial debe haber sido creado por RiverFactory (Steady State)
        assertTrue(passedState.getWaterDepthAt(0) > 0, "El estado inicial debe tener agua (Steady State)");
    }
}