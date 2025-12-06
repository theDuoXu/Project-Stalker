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
 * Test unitario para ManningGpuSolver (Versión Pure Flyweight / Zero-Copy).
 * <p>
 * Verifica:
 * 1. Lazy Initialization.
 * 2. Flujo DMA correcto.
 * 3. Selección de Estrategia.
 * 4. Lógica de Stride y Alineación.
 * 5. RETORNO COMPACTO (Sin expansión innecesaria en esta capa).
 */
class ManningGpuSolverTest {

    private ManningGpuSolver gpuSolver;
    private INativeManningSolver mockNativeSolver;
    private RiverGeometry mockGeometry;

    private final int CELL_COUNT = 50;
    private final long FAKE_SESSION_HANDLE = 12345L;

    @BeforeEach
    void setUp() {
        mockNativeSolver = mock(INativeManningSolver.class);

        // Mock Init
        when(mockNativeSolver.initSession(any(), any(), any(), any(), any(), any(), eq(CELL_COUNT)))
                .thenReturn(FAKE_SESSION_HANDLE);

        mockGeometry = mock(RiverGeometry.class);
        when(mockGeometry.getCellCount()).thenReturn(CELL_COUNT);
        when(mockGeometry.getSpatialResolution()).thenReturn(10.0f);

        float[] dummyArray = new float[CELL_COUNT];
        when(mockGeometry.getBottomWidth()).thenReturn(dummyArray);
        when(mockGeometry.getSideSlope()).thenReturn(dummyArray);
        when(mockGeometry.getManningCoefficient()).thenReturn(dummyArray);
        when(mockGeometry.cloneElevationProfile()).thenReturn(dummyArray);

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
        float[] inflows = {100f};
        float[] dummy = new float[CELL_COUNT];

        when(mockNativeSolver.runBatch(anyLong(), any(), any(), anyInt(), anyInt(), anyInt())).thenReturn(0);

        gpuSolver.solveSmartBatch(dummy, inflows, dummy, true);

        ArgumentCaptor<FloatBuffer> bufferCaptor = ArgumentCaptor.forClass(FloatBuffer.class);
        verify(mockNativeSolver, times(1)).initSession(
                bufferCaptor.capture(), bufferCaptor.capture(), bufferCaptor.capture(),
                bufferCaptor.capture(), bufferCaptor.capture(), bufferCaptor.capture(),
                eq(CELL_COUNT)
        );
        assertEquals(6, bufferCaptor.getAllValues().size());
    }

    @Test
    @DisplayName("Smart Mode: Debe usar MODE_SMART_LAZY y devolver datos COMPACTOS (sin expandir)")
    void solveSmartBatch_shouldUseSmartMode() {
        // --- ARRANGE ---
        int batchSize = 2;
        // ReadWidth será min(batchSize, cellCount) = 2.
        float[] newInflows = {100f, 150f};
        float[] dummy = new float[CELL_COUNT];

        // DATOS COMPACTOS SIMULADOS (Lo que devuelve C++)
        // Tamaño = BatchSize (2) * ReadWidth (2) * Vars (2) = 8 floats.
        int readWidth = 2;
        int compactSize = batchSize * readWidth * 2;
        float[] gpuCompactData = new float[compactSize];

        // Llenamos valores conocidos para verificar posición
        // H: [1.1, 1.2, 1.3, 1.4]
        gpuCompactData[0] = 1.1f;
        gpuCompactData[1] = 1.2f;
        gpuCompactData[2] = 1.3f;

        // V: [2.1, ... ] (Empieza en offset 4)
        gpuCompactData[4] = 2.1f;

        doAnswer(inv -> {
            FloatBuffer out = inv.getArgument(2);
            // Verificamos optimización de buffer
            assertTrue(out.capacity() < 200, "El buffer debe ser pequeño (optimizado)");
            assertTrue(out.capacity() >= compactSize);

            out.clear();
            out.put(gpuCompactData);
            return 0;
        }).when(mockNativeSolver).runBatch(
                eq(FAKE_SESSION_HANDLE), any(), any(), eq(batchSize),
                eq(INativeManningSolver.MODE_SMART_LAZY),
                eq(1)
        );

        // --- ACT ---
        ManningGpuSolver.RawGpuResult result = gpuSolver.solveSmartBatch(dummy, newInflows, dummy, true);

        // --- ASSERT ---
        verify(mockNativeSolver).runBatch(anyLong(), any(), any(), eq(batchSize), eq(INativeManningSolver.MODE_SMART_LAZY), eq(1));

        assertEquals(2, result.storedSteps());
        assertEquals(readWidth, result.activeWidth()); // ActiveWidth debe ser 2

        // 1. Verificamos que NO HUBO EXPANSIÓN (Pure Flyweight)
        // El array debe tener el tamaño compacto (2*2 = 4 floats por variable), NO el ancho completo (2*50=100)
        int expectedSizePerVar = batchSize * readWidth;
        assertEquals(expectedSizePerVar, result.depths().length, "El array devuelto debe ser compacto");

        // 2. Verificamos Integridad de Datos
        assertEquals(1.1f, result.depths()[0], 1e-6f);
        assertEquals(1.2f, result.depths()[1], 1e-6f);
        assertEquals(1.3f, result.depths()[2], 1e-6f);

        assertEquals(2.1f, result.velocities()[0], 1e-6f);
    }

    @Test
    @DisplayName("Full Mode (Stride=1): Debe descargar todo el bloque (ActiveWidth = CellCount)")
    void solveFullEvolutionBatch_Default_shouldDownloadEverything() {
        // --- ARRANGE ---
        int batchSize = 1;
        float[] newInflows = {100f};
        float[] dummy = new float[CELL_COUNT];

        // 1 paso x 50 celdas x 2 vars = 100 floats
        float[] gpuData = new float[CELL_COUNT * 2];
        gpuData[0] = 9.9f;

        doAnswer(inv -> {
            FloatBuffer out = inv.getArgument(2);
            // En Full Mode, readWidth = CellCount. Buffer grande.
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
        ManningGpuSolver.RawGpuResult result = gpuSolver.solveFullEvolutionBatch(dummy, newInflows, dummy);

        // --- ASSERT ---
        assertEquals(1, result.storedSteps());
        assertEquals(CELL_COUNT, result.activeWidth()); // ActiveWidth = Full Width
        assertEquals(CELL_COUNT, result.depths().length);
        assertEquals(9.9f, result.depths()[0], 1e-6f);
    }

    @Test
    @DisplayName("Full Mode (Stride=2): Debe fallar si BatchSize no es múltiplo de Stride")
    void solveFullEvolution_WithBadAlignment_ShouldThrow() {
        int batchSize = 5;
        int stride = 2;
        float[] inflows = new float[batchSize];
        float[] dummy = new float[CELL_COUNT];

        assertThrows(IllegalArgumentException.class, () ->
                gpuSolver.solveFullEvolutionBatch(dummy, inflows, dummy, stride)
        );
    }

    @Test
    @DisplayName("Full Mode (Stride=2): Debe reducir memoria y pasar stride al nativo")
    void solveFullEvolution_WithStride_ShouldReduceMemoryAndOutput() {
        int batchSize = 4;
        int stride = 2;
        int expectedSavedSteps = 2;
        // En Full, readWidth = CellCount
        int expectedTotalFloats = expectedSavedSteps * CELL_COUNT * 2;

        float[] inflows = new float[batchSize];
        float[] dummy = new float[CELL_COUNT];

        doAnswer(inv -> {
            FloatBuffer out = inv.getArgument(2);
            int passedStride = inv.getArgument(5);
            assertEquals(stride, passedStride);
            assertTrue(out.capacity() >= expectedTotalFloats);

            int originalFullSize = batchSize * CELL_COUNT * 2;
            assertTrue(out.capacity() < originalFullSize);
            return 0;
        }).when(mockNativeSolver).runBatch(anyLong(), any(), any(), eq(batchSize), eq(INativeManningSolver.MODE_FULL_EVOLUTION), eq(stride));

        ManningGpuSolver.RawGpuResult result = gpuSolver.solveFullEvolutionBatch(dummy, inflows, dummy, stride);

        assertEquals(expectedSavedSteps, result.storedSteps());
        assertEquals(expectedSavedSteps * CELL_COUNT, result.depths().length);
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