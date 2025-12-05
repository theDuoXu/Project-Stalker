package projectstalker.physics.simulator;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import projectstalker.config.RiverConfig;
import projectstalker.config.SimulationConfig;
import projectstalker.config.SimulationConfig.GpuStrategy; // Única fuente de verdad
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;
import projectstalker.domain.simulation.ISimulationResult;
import projectstalker.factory.RiverGeometryFactory;
import projectstalker.physics.jni.ManningGpuSolver;
import projectstalker.physics.model.RiverPhModel;
import projectstalker.physics.model.RiverTemperatureModel;

import java.lang.reflect.Field;
import java.util.Arrays;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@Slf4j
class ManningBatchProcessorTest {

    private ManningBatchProcessor batchProcessor;
    private RiverGeometry realGeometry;
    private RiverTemperatureModel realTempModel;
    private RiverPhModel realPhModel;
    private SimulationConfig mockConfig;

    // Mocks para pruebas de caja blanca (GPU Strategy)
    private ManningGpuSolver mockGpuSolver;

    private int cellCount;
    private final int BATCH_SIZE = 3;
    private final double DELTA_TIME = 10.0;
    private final int MOCK_STRIDE = 5;

    @BeforeEach
    void setUp() {
        // 1. Geometría y Modelos Reales
        RiverConfig config = RiverConfig.getTestingRiver();
        RiverGeometryFactory factory = new RiverGeometryFactory();
        this.realGeometry = factory.createRealisticRiver(config);
        this.cellCount = this.realGeometry.getCellCount();

        this.realTempModel = new RiverTemperatureModel(config, this.realGeometry);
        this.realPhModel = new RiverPhModel(this.realGeometry);

        // 2. Config Mock
        mockConfig = mock(SimulationConfig.class);
        when(mockConfig.getCpuProcessorCount()).thenReturn(2);
        // Configuramos el stride para verificar que se propaga
        when(mockConfig.getGpuFullEvolutionStride()).thenReturn(MOCK_STRIDE);

        // Default: CPU mode (se sobreescribe en tests específicos)
        when(mockConfig.isUseGpuAccelerationOnManning()).thenReturn(false);

        // 3. Inicializar Processor
        batchProcessor = new ManningBatchProcessor(this.realGeometry, mockConfig);

        // 4. Preparar Mock de GPU Solver (para inyección)
        mockGpuSolver = mock(ManningGpuSolver.class);
    }

    @AfterEach
    void tearDown() {
        if (batchProcessor != null) batchProcessor.close();
    }

    // --- TESTS MODO CPU (Regresión) ---

    @Test
    @DisplayName("CPU Mode: Ejecución normal sin tocar GPU")
    void processBatch_cpuMode_shouldExecuteAndAssembleResults() {
        // Datos de prueba
        RiverState initialState = createSteadyState(0.5f, 1.0f);
        float[] inflows = new float[BATCH_SIZE]; Arrays.fill(inflows, 200f);
        float[][][] phTmp = createDummyPhData();

        // ACT
        ISimulationResult result = batchProcessor.processBatch(
                BATCH_SIZE, initialState, inflows, phTmp,
                false, // GPU Disabled
                GpuStrategy.SMART_SAFE
        );

        // ASSERT
        assertNotNull(result);
        assertEquals(BATCH_SIZE, result.getTimestepCount());

        // Verificación física básica (H aumenta con Q=200 vs Q=50 inicial implícito)
        double initialH = calculateAverageDepth(initialState);
        double finalH = calculateAverageDepth(result.getStateAt(BATCH_SIZE - 1));
        assertTrue(finalH > initialH, "La profundidad debe aumentar en CPU.");
    }

    // --- TESTS MODO GPU (Validación de Estrategias) ---

    @Test
    @DisplayName("Strategy FULL: Debe llamar a solveFullEvolutionBatch pasando el Stride configurado")
    void processBatch_FullStrategy_ShouldCallFullSolverWithStride() throws Exception {
        // Inyectar Mock GPU
        injectMockGpuSolver(batchProcessor, mockGpuSolver);

        // Setup retorno mock para evitar NPE en log
        when(mockGpuSolver.solveFullEvolutionBatch(any(), any(), any(), anyInt()))
                .thenReturn(new float[BATCH_SIZE][2][cellCount]);

        RiverState state = createSteadyState(1f, 1f);
        float[] inflows = new float[BATCH_SIZE];

        // ACT
        batchProcessor.processBatch(
                BATCH_SIZE, state, inflows, createDummyPhData(),
                true, // GPU Enabled
                GpuStrategy.FULL_EVOLUTION
        );

        // ASSERT
        verify(mockGpuSolver).solveFullEvolutionBatch(
                any(), // depths
                any(), // inflows
                any(), // Q
                eq(MOCK_STRIDE) // <--- VERIFICACIÓN CLAVE: Stride propagado
        );
        verify(mockGpuSolver, never()).solveSmartBatch(any(), any(), any(), anyBoolean());
    }

    @Test
    @DisplayName("Strategy SMART_TRUSTED: Debe llamar a solveSmartBatch con trust=true")
    void processBatch_SmartTrusted_ShouldCallSmartSolverTrusted() throws Exception {
        injectMockGpuSolver(batchProcessor, mockGpuSolver);

        when(mockGpuSolver.solveSmartBatch(any(), any(), any(), anyBoolean()))
                .thenReturn(new float[BATCH_SIZE][2][BATCH_SIZE]);

        RiverState state = createSteadyState(1f, 1f);

        // ACT
        batchProcessor.processBatch(
                BATCH_SIZE, state, new float[BATCH_SIZE], createDummyPhData(),
                true,
                GpuStrategy.SMART_TRUSTED
        );

        // ASSERT
        verify(mockGpuSolver).solveSmartBatch(
                any(), any(), any(),
                eq(true) // <--- VERIFICACIÓN CLAVE: Trust = true
        );
    }

    @Test
    @DisplayName("Strategy SMART_SAFE (Fallback): Si Smart falla, debe saltar a Full Evolution")
    void processBatch_SmartSafe_ShouldFallbackToFull_OnException() throws Exception {
        injectMockGpuSolver(batchProcessor, mockGpuSolver);

        // CONFIGURAR FALLO: Smart lanza excepción (Río inestable)
        when(mockGpuSolver.solveSmartBatch(any(), any(), any(), eq(false)))
                .thenThrow(new IllegalStateException("Río inestable detectado"));

        // Configurar recuperación: Full devuelve resultado válido
        when(mockGpuSolver.solveFullEvolutionBatch(any(), any(), any(), anyInt()))
                .thenReturn(new float[BATCH_SIZE][2][cellCount]);

        RiverState state = createSteadyState(1f, 1f);

        // ACT
        batchProcessor.processBatch(
                BATCH_SIZE, state, new float[BATCH_SIZE], createDummyPhData(),
                true,
                GpuStrategy.SMART_SAFE
        );

        // ASSERT
        // 1. Intentó Smart con trust=false
        verify(mockGpuSolver).solveSmartBatch(any(), any(), any(), eq(false));

        // 2. Capturó excepción y llamó a Full con el Stride correcto
        verify(mockGpuSolver).solveFullEvolutionBatch(any(), any(), any(), eq(MOCK_STRIDE));
    }

    // --- Helpers ---

    private void injectMockGpuSolver(ManningBatchProcessor processor, ManningGpuSolver mockSolver) throws Exception {
        Field field = ManningBatchProcessor.class.getDeclaredField("gpuSolver");
        field.setAccessible(true);
        field.set(processor, mockSolver);
    }

    private RiverState createSteadyState(float h, float v) {
        float[] arrH = new float[cellCount]; Arrays.fill(arrH, h);
        float[] arrV = new float[cellCount]; Arrays.fill(arrV, v);
        float[] zeros = new float[cellCount];
        return new RiverState(arrH, arrV, zeros, zeros, zeros);
    }

    private float[][][] createDummyPhData() {
        return new float[BATCH_SIZE][2][cellCount];
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