package projectstalker.physics.jni;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import projectstalker.domain.river.RiverGeometry;

import java.nio.FloatBuffer;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

/**
 * Test unitario para ManningGpuSolver (Versión Stateful DMA / Zero-Copy).
 * Validaciones:
 * 1. Lazy Initialization.
 * 2. DMA Flow (Buffers In/Out).
 * 3. Selección de Estrategia (Smart vs Full).
 * 4. Stride Logic (Muestreo).
 */
class ManningGpuSolverTest {

    private ManningGpuSolver gpuSolver;
    private INativeManningSolver mockNativeSolver;
    private RiverGeometry mockGeometry;

    private final int CELL_COUNT = 50;
    private final long FAKE_SESSION_HANDLE = 12345L;

    @BeforeEach
    void setUp() {
        // 1. Mock del Native Solver
        mockNativeSolver = mock(INativeManningSolver.class);

        // Configurar INIT: Devuelve un handle válido
        when(mockNativeSolver.initSession(any(), any(), any(), any(), any(), any(), eq(CELL_COUNT)))
                .thenReturn(FAKE_SESSION_HANDLE);

        // 2. Mock de Geometría
        mockGeometry = mock(RiverGeometry.class);
        when(mockGeometry.getCellCount()).thenReturn(CELL_COUNT);
        when(mockGeometry.getSpatialResolution()).thenReturn(10.0f);

        float[] dummyArray = new float[CELL_COUNT];
        for(int i=0; i<CELL_COUNT; i++) dummyArray[i] = 1.0f;

        when(mockGeometry.getBottomWidth()).thenReturn(dummyArray);
        when(mockGeometry.getSideSlope()).thenReturn(dummyArray);
        when(mockGeometry.getManningCoefficient()).thenReturn(dummyArray);
        when(mockGeometry.cloneElevationProfile()).thenReturn(dummyArray);

        // 3. SUT
        gpuSolver = new ManningGpuSolver(mockNativeSolver, mockGeometry);
    }

    @AfterEach
    void tearDown() {
        if (gpuSolver != null) {
            try { gpuSolver.close(); } catch (Exception e) {}
        }
    }

    @Test
    @DisplayName("Lazy Init: Primera llamada debe inicializar sesión")
    void solveSmartBatch_shouldTriggerLazyInit() {
        // --- ARRANGE ---
        float[] inflows = {100f};
        float[] dummy = new float[CELL_COUNT];

        // Mock runBatch (con int mode y stride)
        when(mockNativeSolver.runBatch(anyLong(), any(), any(), anyInt(), anyInt(), anyInt())).thenReturn(0);

        // --- ACT ---
        gpuSolver.solveSmartBatch(dummy, inflows, dummy, true);

        // --- ASSERT ---
        ArgumentCaptor<FloatBuffer> bufferCaptor = ArgumentCaptor.forClass(FloatBuffer.class);

        verify(mockNativeSolver, times(1)).initSession(
                bufferCaptor.capture(), bufferCaptor.capture(), bufferCaptor.capture(),
                bufferCaptor.capture(), bufferCaptor.capture(), bufferCaptor.capture(),
                eq(CELL_COUNT)
        );
        assertEquals(6, bufferCaptor.getAllValues().size());
    }

    @Test
    @DisplayName("Smart Mode: Debe usar MODE_SMART_LAZY y Stride=1 implícito")
    void solveSmartBatch_shouldUseSmartMode() {
        // --- ARRANGE ---
        int batchSize = 2;
        float[] newInflows = {100f, 150f};
        float[] dummy = new float[CELL_COUNT];

        float[] gpuData = {
                1.1f, 1.2f, 1.3f, 1.4f, // H (2x2)
                2.1f, 2.2f, 2.3f, 2.4f  // V (2x2)
        };

        doAnswer(inv -> {
            FloatBuffer out = inv.getArgument(2);
            out.clear();
            out.put(gpuData);
            return 0;
        }).when(mockNativeSolver).runBatch(
                eq(FAKE_SESSION_HANDLE), any(), any(), eq(batchSize),
                eq(INativeManningSolver.MODE_SMART_LAZY),
                eq(1) // Stride implícito
        );

        // --- ACT ---
        float[][][] result = gpuSolver.solveSmartBatch(dummy, newInflows, dummy, true);

        // --- ASSERT ---
        verify(mockNativeSolver).runBatch(
                eq(FAKE_SESSION_HANDLE), any(), any(), eq(batchSize),
                eq(INativeManningSolver.MODE_SMART_LAZY),
                eq(1)
        );

        assertEquals(2, result.length); // BatchSize
        assertEquals(2, result[0][0].length); // Width = BatchSize (Triangular)
    }

    @Test
    @DisplayName("Full Mode (Stride=1): Debe usar MODE_FULL_EVOLUTION y bajar todo el rectángulo")
    void solveFullEvolutionBatch_Default_shouldDownloadEverything() {
        // --- ARRANGE ---
        int batchSize = 1;
        float[] newInflows = {100f};
        float[] dummy = new float[CELL_COUNT];

        // 1 paso x 50 celdas
        float[] gpuData = new float[CELL_COUNT * 2];
        gpuData[0] = 9.9f;

        doAnswer(inv -> {
            FloatBuffer out = inv.getArgument(2);
            // Debe caber la matriz completa
            assertTrue(out.capacity() >= gpuData.length);
            out.clear();
            out.put(gpuData);
            return 0;
        }).when(mockNativeSolver).runBatch(
                eq(FAKE_SESSION_HANDLE), any(), any(), eq(batchSize),
                eq(INativeManningSolver.MODE_FULL_EVOLUTION),
                eq(1)
        );

        // --- ACT ---
        float[][][] result = gpuSolver.solveFullEvolutionBatch(dummy, newInflows, dummy);

        // --- ASSERT ---
        assertEquals(1, result.length);
        assertEquals(CELL_COUNT, result[0][0].length); // Width = CellCount
        assertEquals(9.9f, result[0][0][0], 1e-6f);
    }

    @Test
    @DisplayName("Full Mode (Stride=3): Debe reducir memoria y pasar stride al nativo")
    void solveFullEvolution_WithStride_ShouldReduceMemoryAndOutput() {
        // --- ARRANGE ---
        int batchSize = 10;
        int stride = 3;

        // Cálculo esperado de pasos guardados: ceil(10/3) = 4
        // Pasos: 0, 3, 6, 9.
        int expectedSavedSteps = 4;
        int expectedTotalFloats = expectedSavedSteps * CELL_COUNT * 2; // H y V

        float[] inflows = new float[batchSize];
        float[] dummy = new float[CELL_COUNT];

        // Mock para verificar tamaño de buffer
        doAnswer(inv -> {
            FloatBuffer out = inv.getArgument(2);
            int passedStride = inv.getArgument(5);

            assertEquals(stride, passedStride, "El Stride debe llegar al JNI");

            // VERIFICACIÓN CRÍTICA DE MEMORIA
            // El buffer debe tener el tamaño reducido, no el tamaño 'batchSize * CellCount * 2'
            assertEquals(expectedTotalFloats, out.capacity(), "El buffer Output debe estar dimensionado para los pasos reducidos");

            return 0;
        }).when(mockNativeSolver).runBatch(
                eq(FAKE_SESSION_HANDLE), any(), any(), eq(batchSize),
                eq(INativeManningSolver.MODE_FULL_EVOLUTION),
                eq(stride)
        );

        // --- ACT ---
        float[][][] result = gpuSolver.solveFullEvolutionBatch(dummy, inflows, dummy, stride);

        // --- ASSERT ---
        verify(mockNativeSolver).runBatch(anyLong(), any(), any(), eq(batchSize), eq(1), eq(stride));

        // Verificar dimensión temporal del resultado Java
        assertEquals(expectedSavedSteps, result.length, "El resultado debe tener solo los pasos guardados");
        assertEquals(CELL_COUNT, result[0][0].length, "El ancho debe ser el río completo");
    }

    @Test
    @DisplayName("Validation: solveSmartBatch sin trust debe lanzar excepción si el río es inestable")
    void solveSmartBatch_shouldThrowIfUnstable() {
        float[] inflows = {100f};
        float[] dummy = new float[CELL_COUNT];

        float[] unstableQ = new float[CELL_COUNT];
        unstableQ[2] = 10f;
        unstableQ[CELL_COUNT-3] = 100f;

        assertThrows(IllegalStateException.class, () ->
                gpuSolver.solveSmartBatch(dummy, inflows, unstableQ, false)
        );
    }
}