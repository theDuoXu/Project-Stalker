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
 * 4. Sanitización.
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

        // Mock runBatch (con int mode)
        when(mockNativeSolver.runBatch(anyLong(), any(), any(), anyInt(), anyInt())).thenReturn(0);

        // --- ACT ---
        // Usamos trust=true para saltar validación de steady state y forzar ejecución directa
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
    @DisplayName("Smart Mode: Debe llamar a nativo con MODE_SMART_LAZY y desempaquetar triángulo")
    void solveSmartBatch_shouldUseSmartMode() {
        // --- ARRANGE ---
        int batchSize = 2;
        float[] newInflows = {100f, 150f};
        float[] dummy = new float[CELL_COUNT];

        // Output simulado (Smart = Triangular/Cuadrado Batch x Batch)
        // 2x2 = 4 celdas -> 8 floats
        float[] gpuData = {
                1.1f, 1.2f, 1.3f, 1.4f, // H
                2.1f, 2.2f, 2.3f, 2.4f  // V
        };

        doAnswer(inv -> {
            FloatBuffer out = inv.getArgument(2);
            out.clear();
            out.put(gpuData);
            return 0;
        }).when(mockNativeSolver).runBatch(
                eq(FAKE_SESSION_HANDLE), any(), any(), eq(batchSize),
                eq(INativeManningSolver.MODE_SMART_LAZY) // <--- VERIFICACIÓN CRÍTICA DEL MODO
        );

        // --- ACT ---
        float[][][] result = gpuSolver.solveSmartBatch(dummy, newInflows, dummy, true);

        // --- ASSERT ---
        verify(mockNativeSolver).runBatch(
                eq(FAKE_SESSION_HANDLE), any(), any(), eq(batchSize),
                eq(INativeManningSolver.MODE_SMART_LAZY)
        );

        // Verificar desempaquetado (Ancho = BatchSize = 2)
        assertEquals(2, result.length);
        assertEquals(2, result[0][0].length); // Width es 2, no 50
        assertEquals(1.1f, result[0][0][0], 1e-6f);
    }

    @Test
    @DisplayName("Full Evolution Mode: Debe llamar a nativo con MODE_FULL_EVOLUTION y desempaquetar todo")
    void solveFullEvolutionBatch_shouldUseFullMode() {
        // --- ARRANGE ---
        int batchSize = 1; // 1 paso
        float[] newInflows = {100f};
        float[] dummy = new float[CELL_COUNT];

        // Output simulado (Full = Batch x CellCount)
        // 1 x 50 = 50 celdas -> 100 floats
        float[] gpuData = new float[CELL_COUNT * 2];
        gpuData[0] = 9.9f; // H en celda 0
        gpuData[CELL_COUNT] = 8.8f; // V en celda 0 (Offset = 50)

        doAnswer(inv -> {
            FloatBuffer out = inv.getArgument(2);
            // Verificamos que el buffer reservado sea gigante
            assertTrue(out.capacity() >= gpuData.length);
            out.clear();
            out.put(gpuData);
            return 0;
        }).when(mockNativeSolver).runBatch(
                eq(FAKE_SESSION_HANDLE), any(), any(), eq(batchSize),
                eq(INativeManningSolver.MODE_FULL_EVOLUTION) // <--- VERIFICACIÓN CRÍTICA DEL MODO
        );

        // --- ACT ---
        float[][][] result = gpuSolver.solveFullEvolutionBatch(dummy, newInflows, dummy);

        // --- ASSERT ---
        verify(mockNativeSolver).runBatch(
                eq(FAKE_SESSION_HANDLE), any(), any(), eq(batchSize),
                eq(INativeManningSolver.MODE_FULL_EVOLUTION)
        );

        // Verificar desempaquetado (Ancho = CellCount = 50)
        assertEquals(1, result.length);
        assertEquals(CELL_COUNT, result[0][0].length); // Width es 50
        assertEquals(9.9f, result[0][0][0], 1e-6f);
        assertEquals(8.8f, result[0][1][0], 1e-6f);
    }

    @Test
    @DisplayName("Validation: solveSmartBatch sin trust debe lanzar excepción si el río es inestable")
    void solveSmartBatch_shouldThrowIfUnstable() {
        float[] inflows = {100f};
        float[] dummy = new float[CELL_COUNT];

        // Caudal inestable: Empieza en 10, acaba en 100
        float[] unstableQ = new float[CELL_COUNT];
        unstableQ[2] = 10f;
        unstableQ[CELL_COUNT-3] = 100f;

        assertThrows(IllegalStateException.class, () ->
                gpuSolver.solveSmartBatch(dummy, inflows, unstableQ, false) // trust=false
        );
    }
}