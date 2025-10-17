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
import projectstalker.physics.impl.SequentialManningHydrologySolver;
import projectstalker.physics.jni.ManningGpuSolver;

import java.lang.reflect.Field;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

/**
 * Test unitario para ManningSimulator.
 * Se centra en la correcta orquestación (advanceTimeStep) y la delegación al BatchProcessor.
 * Mockea todas las clases necesarias para permitir aislamiento total sobre la implementación ya que
 * esta clase es principalmente un orquestador.
 */
@Slf4j
class ManningSimulatorTest {

    private ManningSimulator simulator;
    private RiverConfig mockConfig;
    private SimulationConfig mockSimConfig;
    private RiverGeometry mockGeometry;
    private ManningBatchProcessor mockBatchProcessor; // Mock para delegación

    private final int CELL_COUNT = 5;
    private final double DELTA_TIME = 10.0;

    @BeforeEach
    void setUp() throws NoSuchFieldException, IllegalAccessException {
        // --- 1. Mocks Base ---
        mockConfig = mock(RiverConfig.class);

        // Parámetros usados para calcular CELL_COUNT (25.0 / 5.0 = 5)
        when(mockConfig.totalLength()).thenReturn(25.0);
        when(mockConfig.spatialResolution()).thenReturn(5.0);

        // Otros parámetros usados en RiverGeometryFactory.createRealisticRiver()
        when(mockConfig.seed()).thenReturn(12345L);
        when(mockConfig.initialElevation()).thenReturn(100.0);
        when(mockConfig.concavityFactor()).thenReturn(0.0);
        when(mockConfig.averageSlope()).thenReturn(0.001);
        when(mockConfig.slopeVariability()).thenReturn(0.0);
        when(mockConfig.baseWidth()).thenReturn(10.0);
        when(mockConfig.widthVariability()).thenReturn(0.0);
        when(mockConfig.baseSideSlope()).thenReturn(2.0);
        when(mockConfig.sideSlopeVariability()).thenReturn(0.0);
        when(mockConfig.baseManning()).thenReturn(0.035);
        when(mockConfig.manningVariability()).thenReturn(0.0);
        when(mockConfig.baseDecayRateAt20C()).thenReturn(0.1);
        when(mockConfig.decayRateVariability()).thenReturn(0.0);
        when(mockConfig.basePh()).thenReturn(7.5); // Necesario para RiverPhModel
        when(mockConfig.phVariability()).thenReturn(0.0);
        when(mockConfig.baseTemperature()).thenReturn(15.0); // Necesario para RiverTemperatureModel
        when(mockConfig.averageAnnualTemperature()).thenReturn(15.0);
        when(mockConfig.dailyTempVariation()).thenReturn(0.0);
        when(mockConfig.seasonalTempVariation()).thenReturn(0.0);
        when(mockConfig.maxHeadwaterCoolingEffect()).thenReturn(0.0);
        when(mockConfig.headwaterCoolingDistance()).thenReturn(1.0);
        when(mockConfig.widthHeatingFactor()).thenReturn(0.0);
        when(mockConfig.slopeCoolingFactor()).thenReturn(0.0);
        when(mockConfig.temperatureNoiseAmplitude()).thenReturn(0.0);
        when(mockConfig.noiseFrequency()).thenReturn(0.05f); // Necesario para FastNoiseLite
        when(mockConfig.detailNoiseFrequency()).thenReturn(0.05f);
        when(mockConfig.zoneNoiseFrequency()).thenReturn(0.001f);


        mockSimConfig = mock(SimulationConfig.class);
        // Simular un FlowConfig simple para evitar NPE en FlowProfileModel
        SimulationConfig.FlowConfig mockFlowConfig = mock(SimulationConfig.FlowConfig.class);
        when(mockFlowConfig.getBaseDischarge()).thenReturn(10.0);
        when(mockFlowConfig.getNoiseAmplitude()).thenReturn(5.0);
        when(mockFlowConfig.getNoiseFrequency()).thenReturn(0.001f);
        when(mockSimConfig.getFlowConfig()).thenReturn(mockFlowConfig);

        when(mockSimConfig.isUseGpuAccelerationOnManning()).thenReturn(false);

        // 2. Simulación de Geometría Mockeada (para el set del simulador)
        // La geometría real es creada en la línea 65 de ManningSimulator.
        // Aquí creamos una geometría mockeada para inyectar después y controlar los getters
        mockGeometry = mock(RiverGeometry.class);
        when(mockGeometry.getCellCount()).thenReturn(CELL_COUNT);
        when(mockGeometry.clonePhProfile()).thenReturn(new double[CELL_COUNT]);

        // --- 3. Inicializar el Simulador y Mocks de Dependencias Pesadas ---

        // Llama al constructor REAL. Ahora no fallará porque mockConfig tiene los valores geométricos.
        simulator = new ManningSimulator(mockConfig, mockSimConfig);

        // ********** SUSTITUCIÓN DE DEPENDENCIAS CON REFLEXIÓN **********
        // Sustituimos las instancias creadas INTERNAMENTE por mocks para aislar el test.

        // 3.1 Inyectar Mock de Geometría (para que el estado tenga el tamaño correcto)
        Field geometryField = ManningSimulator.class.getDeclaredField("geometry");
        geometryField.setAccessible(true);
        geometryField.set(simulator, mockGeometry);

        // 3.2 Inyectar Mock de BatchProcessor (para probar la delegación)
        mockBatchProcessor = mock(ManningBatchProcessor.class);
        Field batchProcessorField = ManningSimulator.class.getDeclaredField("batchProcessor");
        batchProcessorField.setAccessible(true);
        batchProcessorField.set(simulator, mockBatchProcessor);

        // 3.3 Inyectar Mock de CpuSolver (para simular resultados en modo paso a paso)
        SequentialManningHydrologySolver mockCpuSolver = mock(SequentialManningHydrologySolver.class);
        Field cpuSolverField = ManningSimulator.class.getDeclaredField("cpuSolver");
        cpuSolverField.setAccessible(true);
        cpuSolverField.set(simulator, mockCpuSolver);

        // Configurar el mock del CPU Solver para devolver un estado de "éxito"
        double[] successData = new double[CELL_COUNT];
        Arrays.fill(successData, 1.0);
        // Nota: Los arrays deben ser de longitud 5 (CELL_COUNT)
        RiverState mockNextStateHydro = new RiverState(successData.clone(), successData.clone(), successData.clone(), successData.clone());
        when(mockCpuSolver.calculateNextState(any(), any(), any(), anyDouble(), anyDouble())).thenReturn(mockNextStateHydro);

        // --- 4. Estado Inicial del Simulador ---
        simulator.setCurrentState(new RiverState(new double[CELL_COUNT], new double[CELL_COUNT], new double[CELL_COUNT], new double[CELL_COUNT]));
        simulator.setCurrentTimeInSeconds(100.0);
        simulator.setGpuAccelerated(false);
        log.info("ManningSimulator inicializado para pruebas de orquestación con {} celdas simuladas.", CELL_COUNT);
    }

    // El método `calculateTemperatureAndPh()` ya ha sido probado indirectamente en RiverPhTest
    // y se asume que funciona, ya que usa modelos reales.

    // --------------------------------------------------------------------------
    // TEST 1: Avance de Paso Único (Modo CPU)
    // --------------------------------------------------------------------------

    @Test
    @DisplayName("advanceTimeStep (CPU) debe avanzar el tiempo y delegar al cpuSolver")
    void advanceTimeStep_cpuMode_shouldAdvanceTimeAndUseCpuSolver() throws NoSuchFieldException, IllegalAccessException {
        log.info("Ejecutando test: advanceTimeStep en modo CPU.");

        double initialTime = simulator.getCurrentTimeInSeconds();

        // ACT
        simulator.advanceTimeStep(DELTA_TIME);

        // ASSERT 1: Flujo de control y Estado
        assertEquals(initialTime + DELTA_TIME, simulator.getCurrentTimeInSeconds(), 0.001, "El tiempo debe avanzar.");

        // Recuperar el mock del cpuSolver
        Field cpuSolverField = ManningSimulator.class.getDeclaredField("cpuSolver");
        cpuSolverField.setAccessible(true);
        SequentialManningHydrologySolver mockSolver = (SequentialManningHydrologySolver) cpuSolverField.get(simulator);

        // ASSERT 2: Verificación de Delegación
        verify(mockSolver, times(1)).calculateNextState(any(), any(), any(), anyDouble(), anyDouble());

        // ASSERT 3: Estado Final
        assertEquals(1.0, simulator.getCurrentState().getWaterDepthAt(0), 0.001, "El estado debe ser actualizado por el mockSolver.");
        assertTrue(simulator.getCpuFillIterations() > 0, "Las métricas deben actualizarse.");

        log.info("Paso único CPU verificado. Tiempo final: {}", simulator.getCurrentTimeInSeconds());
    }

    // --------------------------------------------------------------------------
    // TEST 2: Avance de Paso Único (Modo GPU)
    // --------------------------------------------------------------------------

    @Test
    @DisplayName("advanceTimeStep (GPU) debe delegar al ManningGpuSolver nativo")
    void advanceTimeStep_gpuMode_shouldDelegateToGpuSolver() {
        log.info("Ejecutando test: advanceTimeStep en modo GPU.");
        simulator.setGpuAccelerated(true);
        double initialTime = simulator.getCurrentTimeInSeconds();

        // 1. Mockear la llamada estática nativa de ManningGpuSolver
        // Nota: Esto requiere la librería Mockito `mock-static` para simular llamadas estáticas (JNI)

        // Simular que el GPU Solver devuelve un resultado con profundidad 5.0 y velocidad 2.0
        double[] gpuDepthResult = new double[CELL_COUNT]; Arrays.fill(gpuDepthResult, 5.0);
        double[] gpuVelocityResult = new double[CELL_COUNT]; Arrays.fill(gpuVelocityResult, 2.0);
        double[][] mockGpuResults = new double[][]{gpuDepthResult, gpuVelocityResult};

        try (var mocked = mockStatic(ManningGpuSolver.class)) {
            mocked.when(() -> ManningGpuSolver.solve(any(), any(), any(), anyDouble())).thenReturn(mockGpuResults);

            // ACT
            simulator.advanceTimeStep(DELTA_TIME);

            // ASSERT 1: Delegación
            mocked.verify(() -> ManningGpuSolver.solve(any(), any(), any(), eq(initialTime)), times(1));

            // ASSERT 2: Estado Final
            assertEquals(5.0, simulator.getCurrentState().getWaterDepthAt(0), 0.001, "El estado debe ser actualizado por el mock GPU.");
        }

        assertEquals(initialTime + DELTA_TIME, simulator.getCurrentTimeInSeconds(), 0.001, "El tiempo debe avanzar.");
        log.info("Paso único GPU verificado. El estado fue actualizado a profundidad 5.0.");
    }

    // --------------------------------------------------------------------------
    // TEST 3: Avance por Lote (Batch)
    // --------------------------------------------------------------------------

    @Test
    @DisplayName("advanceBatchTimeStep debe delegar la ejecución completa al BatchProcessor")
    void advanceBatchTimeStep_shouldDelegateToBatchProcessor() throws Exception {
        log.info("Ejecutando test: advanceBatchTimeStep.");
        int batchSize = 3;
        double initialTime = simulator.getCurrentTimeInSeconds();

        // 1. Configurar el Mock de BatchProcessor para devolver un resultado final
        double[] finalDepth = new double[CELL_COUNT]; Arrays.fill(finalDepth, 10.0);
        RiverState finalState = new RiverState(finalDepth, new double[CELL_COUNT], new double[CELL_COUNT], new double[CELL_COUNT]);

        // Simular un ManningSimulationResult con los 3 pasos, donde el último es el finalState
        ManningSimulationResult mockResult = ManningSimulationResult.builder()
                .geometry(mockGeometry)
                .states(List.of(mock(RiverState.class), mock(RiverState.class), finalState))
                .build();

        // MOCKEAR LA SECUENCIA DE LLAMADAS DEL BATCH PROCESSOR
        // El simulador llama a createDischargeProfiles y luego a processBatch
        double[][] mockDischargeProfiles = new double[batchSize][CELL_COUNT];
        when(mockBatchProcessor.createDischargeProfiles(anyInt(), any(), any())).thenReturn(mockDischargeProfiles);
        when(mockBatchProcessor.processBatch(anyInt(), anyDouble(), any(), any(), any(), anyBoolean())).thenReturn(mockResult);

        // ACT
        ManningSimulationResult result = simulator.advanceBatchTimeStep(DELTA_TIME, batchSize);

        // ASSERT 1: Delegación
        verify(mockBatchProcessor, times(1)).createDischargeProfiles(eq(batchSize), any(), any());
        verify(mockBatchProcessor, times(1)).processBatch(eq(batchSize), eq(initialTime), any(), eq(mockDischargeProfiles), any(), eq(false));

        // ASSERT 2: Flujo de Control
        assertEquals(mockResult, result, "El resultado devuelto debe ser el del BatchProcessor.");

        // ASSERT 3: Actualización del Estado Final
        assertEquals(initialTime + (batchSize * DELTA_TIME), simulator.getCurrentTimeInSeconds(), 0.001, "El tiempo debe avanzar por el tamaño completo del batch.");
        assertEquals(10.0, simulator.getCurrentState().getWaterDepthAt(0), 0.001, "El estado final debe ser actualizado al último estado del batch.");

        log.info("Avance por lote verificado. Delegación y actualización del estado final correctas.");
    }
}