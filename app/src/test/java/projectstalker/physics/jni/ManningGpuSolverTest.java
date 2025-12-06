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
 * Test unitario para ManningGpuSolver.
 * <p>
 * Verifica:
 * 1. Lazy Initialization.
 * 2. Flujo DMA correcto.
 * 3. Selección de Estrategia.
 * 4. Lógica de Stride y Alineación.
 * 5. Expansión Inteligente (Unpacking) de datos compactos a full-width.
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
    @DisplayName("Smart Mode: Debe usar MODE_SMART_LAZY y expandir datos compactos a full width")
    void solveSmartBatch_shouldUseSmartMode() {
        // --- ARRANGE ---
        int batchSize = 2; // ReadWidth será min(2, 50) = 2.
        float[] newInflows = {100f, 150f};
        float[] dummy = new float[CELL_COUNT];

        // DATOS COMPACTOS SIMULADOS (Lo que devuelve C++)
        // Tamaño = BatchSize (2) * ReadWidth (2) * Vars (2) = 8 floats.
        // Bloque H (4 floats): [H_t0_c0, H_t0_c1, H_t1_c0, H_t1_c1]
        // Bloque V (4 floats): [V_t0_c0, V_t0_c1, V_t1_c0, V_t1_c1]

        int readWidth = 2;
        int compactSize = batchSize * readWidth * 2;
        float[] gpuCompactData = new float[compactSize];

        // Llenamos valores conocidos para verificar posición
        gpuCompactData[0] = 1.1f; // H[t=0, c=0]
        gpuCompactData[1] = 1.2f; // H[t=0, c=1]
        gpuCompactData[2] = 1.3f; // H[t=1, c=0]

        // Bloque V empieza en index 4
        gpuCompactData[4] = 2.1f; // V[t=0, c=0]

        doAnswer(inv -> {
            FloatBuffer out = inv.getArgument(2);

            // Verificación crítica: El buffer que pasa Java debe ser PEQUEÑO (optimizado)
            // No debe ser de tamaño 200 (Batch * CellCount * 2)
            assertTrue(out.capacity() < 200, "El buffer debe estar dimensionado para datos compactos");
            assertTrue(out.capacity() >= compactSize, "El buffer debe tener espacio suficiente");

            out.clear();
            out.put(gpuCompactData);
            return 0;
        }).when(mockNativeSolver).runBatch(
                eq(FAKE_SESSION_HANDLE), any(), any(), eq(batchSize),
                eq(INativeManningSolver.MODE_SMART_LAZY),
                eq(1)
        );

        // --- ACT ---
        // Java leerá los 8 floats y los expandirá a matrices de ancho 50 (rellenando ceros)
        ManningGpuSolver.RawGpuResult result = gpuSolver.solveSmartBatch(dummy, newInflows, dummy, true);

        // --- ASSERT ---
        verify(mockNativeSolver).runBatch(anyLong(), any(), any(), eq(batchSize), eq(INativeManningSolver.MODE_SMART_LAZY), eq(1));

        assertEquals(2, result.storedSteps());
        assertEquals(2, result.activeWidth()); // readWidth era 2
        // 1. Verificamos EXPANSIÓN
        // El array resultante debe tener el ancho completo del río (50 celdas)
        assertEquals(batchSize * CELL_COUNT, result.depths().length);

        // 2. Verificamos POSICIONAMIENTO
        // t=0, c=0 -> Index 0
        assertEquals(1.1f, result.depths()[0], 1e-6f);
        // t=0, c=1 -> Index 1
        assertEquals(1.2f, result.depths()[1], 1e-6f);

        // t=1, c=0 -> Index 50 (Inicio de la segunda fila completa)
        // Si no hubiera expansión, esto estaría en index 2 (compacto).
        assertEquals(1.3f, result.depths()[CELL_COUNT], 1e-6f);

        // 3. Verificamos PADDING (Ceros)
        // t=0, c=5 (Fuera del triángulo activo) debe ser 0.0
        assertEquals(0.0f, result.depths()[5], 1e-6f);

        // Verificamos Velocidad t=0, c=0
        assertEquals(2.1f, result.velocities()[0], 1e-6f);
    }

    @Test
    @DisplayName("Full Mode (Stride=1): Debe descargar todo el bloque")
    void solveFullEvolutionBatch_Default_shouldDownloadEverything() {
        // --- ARRANGE ---
        int batchSize = 1;
        float[] newInflows = {100f};
        float[] dummy = new float[CELL_COUNT];

        // 1 paso x 50 celdas x 2 vars = 100 floats (Full Width)
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

            // Verificamos reducción vs full size sin stride
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