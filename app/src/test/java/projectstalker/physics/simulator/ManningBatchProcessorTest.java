package projectstalker.physics.simulator;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import projectstalker.config.RiverConfig;
import projectstalker.config.SimulationConfig;
import projectstalker.config.SimulationConfig.GpuStrategy;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;
import projectstalker.domain.simulation.ChunkedManningResult;
import projectstalker.domain.simulation.IManningResult;
import projectstalker.domain.simulation.StridedManningResult;
import projectstalker.factory.RiverGeometryFactory;
import projectstalker.physics.jni.ManningGpuSolver;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@Slf4j
class ManningBatchProcessorTest {

    private ManningBatchProcessor batchProcessor;
    private RiverGeometry realGeometry;
    private SimulationConfig mockConfig;
    private ManningGpuSolver mockGpuSolver;

    private int cellCount;
    // CONFIGURACIÓN VÁLIDA POR DEFECTO: Batch múltiplo de Stride
    private final int MOCK_STRIDE = 4;
    private final int BATCH_SIZE_CONFIG = 12; // 12 % 4 == 0 (OK)

    @BeforeEach
    void setUp() throws Exception {
        RiverConfig config = RiverConfig.getTestingRiver();
        this.realGeometry = new RiverGeometryFactory().createRealisticRiver(config);
        this.cellCount = this.realGeometry.getCellCount();

        mockConfig = mock(SimulationConfig.class);
        when(mockConfig.getCpuProcessorCount()).thenReturn(2);

        when(mockConfig.getGpuFullEvolutionStride()).thenReturn(MOCK_STRIDE);
        when(mockConfig.getCpuTimeBatchSize()).thenReturn(BATCH_SIZE_CONFIG);

        when(mockConfig.isUseGpuAccelerationOnManning()).thenReturn(true);
        when(mockConfig.getGpuStrategy()).thenReturn(GpuStrategy.FULL_EVOLUTION);

        mockGpuSolver = mock(ManningGpuSolver.class);

        // Usamos la subclase para evitar UnsatisfiedLinkError
        batchProcessor = new TestableManningBatchProcessor(this.realGeometry, mockConfig, mockGpuSolver);
    }

    @AfterEach
    void tearDown() {
        if (batchProcessor != null) batchProcessor.close();
    }

    @Test
    @DisplayName("Padding Logic: Input final irregular debe rellenarse para cumplir Stride")
    void process_FullEvolution_ShouldApplyPadding() {
        // ARRANGE
        float[] inputs = new float[15];
        RiverState state = createSteadyState(1f, 1f);

        // CORRECCIÓN: Añadido 4º argumento (activeWidth) al constructor de RawGpuResult
        // En Full Evolution, activeWidth = cellCount
        when(mockGpuSolver.solveFullEvolutionBatch(any(), any(), any(), anyInt()))
                .thenReturn(new ManningGpuSolver.RawGpuResult(new float[0], new float[0], 0, cellCount));

        // ACT
        batchProcessor.process(inputs, state);

        // ASSERT
        // Verificamos que se llamó al solver con un array de tamaño 4 para el último lote
        verify(mockGpuSolver, atLeastOnce()).solveFullEvolutionBatch(
                any(),
                argThat(arr -> arr.length == 4), // Verifica que llegó el lote con padding
                any(),
                eq(4)
        );
    }

    @Test
    @DisplayName("Result Factory: Debe retornar Strided si cabe en memoria")
    void process_SmallData_ShouldReturnStridedResult() {
        // Input: 24 pasos (2 lotes exactos de 12)
        float[] inputs = new float[24];
        RiverState state = createSteadyState(1f, 1f);

        int storedSteps = 6; // Simulado (24/4 = 6)
        float[] rawData = new float[storedSteps * cellCount];

        // CORRECCIÓN: Añadido activeWidth
        when(mockGpuSolver.solveFullEvolutionBatch(any(), any(), any(), anyInt()))
                .thenReturn(new ManningGpuSolver.RawGpuResult(rawData, rawData, storedSteps / 2, cellCount)); // storedSteps per batch = 3

        // ACT
        IManningResult result = batchProcessor.process(inputs, state);

        // ASSERT
        assertTrue(result instanceof StridedManningResult);
        assertEquals(inputs.length, result.getTimestepCount());
    }

    @Test
    @DisplayName("Smart Fallback: Debe cambiar a Full Evolution si Smart falla")
    void process_SmartFallback_ShouldCallFullEvolution() {
        when(mockConfig.getGpuStrategy()).thenReturn(GpuStrategy.SMART_SAFE);

        RiverState state = createSteadyState(1f, 1f);
        float[] inputs = new float[12];

        // 1. Simular fallo
        when(mockGpuSolver.solveSmartBatch(any(), any(), any(), eq(false)))
                .thenThrow(new IllegalStateException("Unstable"));

        // 2. Simular éxito en Full
        // CORRECCIÓN: Añadido activeWidth
        when(mockGpuSolver.solveFullEvolutionBatch(any(), any(), any(), anyInt()))
                .thenReturn(new ManningGpuSolver.RawGpuResult(new float[0], new float[0], 0, cellCount));

        // ACT
        batchProcessor.process(inputs, state);

        // ASSERT
        verify(mockGpuSolver).solveSmartBatch(any(), any(), any(), eq(false));
        verify(mockGpuSolver).solveFullEvolutionBatch(any(), any(), any(), anyInt());
    }

    // --- Helpers ---
    private RiverState createSteadyState(float h, float v) {
        float[] arr = new float[cellCount]; Arrays.fill(arr, h);
        return new RiverState(arr, arr, arr, arr, arr);
    }

    // --- SUBCLASE CRÍTICA PARA TESTEAR SIN JNI ---
    static class TestableManningBatchProcessor extends ManningBatchProcessor {
        private final ManningGpuSolver mockSolver;

        public TestableManningBatchProcessor(RiverGeometry geo, SimulationConfig conf, ManningGpuSolver mockSolver) {
            super(geo, conf);
            this.mockSolver = mockSolver;
        }

        @Override
        protected ManningGpuSolver createGpuSolver(RiverGeometry geometry) {
            return mockSolver;
        }
    }
}