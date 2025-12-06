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
 * Test unitario para ManningGpuSolver (Versión Stateful DMA / Zero-Copy / Raw Result).
 * <p>
 * Validaciones Actualizadas:
 * 1. Lazy Initialization.
 * 2. DMA Flow (Buffers In/Out).
 * 3. Selección de Estrategia (Smart vs Full).
 * 4. Stride Logic (Alineación estricta y reducción de memoria).
 * 5. Formato de Salida (RawGpuResult).
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
    @DisplayName("Smart Mode: Debe usar MODE_SMART_LAZY y devolver RawGpuResult expandido")
    void solveSmartBatch_shouldUseSmartMode() {
        // --- ARRANGE ---
        int batchSize = 2;
        float[] newInflows = {100f, 150f};
        float[] dummy = new float[CELL_COUNT];

        // El mock debe devolver datos COMPACTOS, tal como lo hace C++.
        // En Smart Mode, readWidth = min(batchSize, cellCount) = 2.
        // Tamaño Buffer = batchSize * readWidth * 2 vars = 2 * 2 * 2 = 8 floats.

        int readWidth = 2;
        int compactSize = batchSize * readWidth * 2;
        float[] gpuCompactData = new float[compactSize];

        // Llenamos datos simulados en formato SoA Compacto:
        // Bloque H (4 floats): [H_t0_c0, H_t0_c1, H_t1_c0, H_t1_c1]
        // Bloque V (4 floats): [V_t0_c0, V_t0_c1, V_t1_c0, V_t1_c1]

        // H en t=0, c=0
        gpuCompactData[0] = 1.1f;

        // V en t=0, c=0. Offset = Tamaño bloque H (Batch * ReadWidth) = 4
        gpuCompactData[4] = 2.1f;

        doAnswer(inv -> {
            FloatBuffer out = inv.getArgument(2);
            // Verificamos que el buffer real sea pequeño (optimizado)
            // Capacidad debe ser aprox 8 (con el margen 1.2x puede ser 9 o 10)
            assertTrue(out.capacity() < 200, "El buffer debería estar optimizado para Smart Mode");

            out.clear();
            out.put(gpuCompactData); // Ahora cabe perfectamente
            return 0;
        }).when(mockNativeSolver).runBatch(
                eq(FAKE_SESSION_HANDLE), any(), any(), eq(batchSize),
                eq(INativeManningSolver.MODE_SMART_LAZY),
                eq(1)
        );

        // --- ACT ---
        // Java leerá los 8 floats y los expandirá a 200 floats (rellenando ceros)
        ManningGpuSolver.RawGpuResult result = gpuSolver.solveSmartBatch(dummy, newInflows, dummy, true);

        // --- ASSERT ---
        verify(mockNativeSolver).runBatch(
                eq(FAKE_SESSION_HANDLE), any(), any(), eq(batchSize),
                eq(INativeManningSolver.MODE_SMART_LAZY),
                eq(1)
        );

        assertEquals(2, result.storedSteps());

        // Verificamos que Java HIZO LA EXPANSIÓN correctamente
        // El array resultante debe ser FULL WIDTH (50 celdas)
        assertEquals(2 * CELL_COUNT, result.depths().length);

        // Verificamos que los datos se colocaron en el índice correcto
        // t=0, c=0 está en índice 0
        assertEquals(1.1f, result.depths()[0], 1e-6f);
        assertEquals(2.1f, result.velocities()[0], 1e-6f);

        // Verificamos que el resto está a cero (padding de expansión)
        // t=0, c=5 (fuera del triángulo activo) debe ser 0
        assertEquals(0.0f, result.depths()[5], 1e-6f);
    }

    @Test
    @DisplayName("Full Mode (Stride=1): Debe descargar todo el bloque")
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
        // --- ARRANGE ---
        int batchSize = 5; // Impar
        int stride = 2;    // Par
        float[] inflows = new float[batchSize];
        float[] dummy = new float[CELL_COUNT];

        // --- ACT & ASSERT ---
        assertThrows(IllegalArgumentException.class, () ->
                gpuSolver.solveFullEvolutionBatch(dummy, inflows, dummy, stride)
        );
    }

    @Test
    @DisplayName("Full Mode (Stride=2): Debe reducir memoria y pasar stride al nativo")
    void solveFullEvolution_WithStride_ShouldReduceMemoryAndOutput() {
        // --- ARRANGE ---
        int batchSize = 4; // Múltiplo de 2
        int stride = 2;

        // Pasos guardados: 4 / 2 = 2.
        int expectedSavedSteps = 2;
        int expectedTotalFloats = expectedSavedSteps * CELL_COUNT * 2;

        float[] inflows = new float[batchSize];
        float[] dummy = new float[CELL_COUNT];

        // Mock para verificar tamaño de buffer
        doAnswer(inv -> {
            FloatBuffer out = inv.getArgument(2);
            int passedStride = inv.getArgument(5);

            assertEquals(stride, passedStride, "El Stride debe llegar al JNI");

            // Verificamos capacidad suficiente para 2 pasos
            assertTrue(out.capacity() >= expectedTotalFloats);

            // Verificamos REDUCCIÓN: Buffer debe ser menor que lo necesario para 4 pasos
            int originalFullSize = batchSize * CELL_COUNT * 2;
            assertTrue(out.capacity() < originalFullSize,
                    "El buffer debería ser menor que el tamaño Full sin stride. Capacidad: " + out.capacity());

            return 0;
        }).when(mockNativeSolver).runBatch(
                eq(FAKE_SESSION_HANDLE), any(), any(), eq(batchSize),
                eq(INativeManningSolver.MODE_FULL_EVOLUTION),
                eq(stride)
        );

        // --- ACT ---
        ManningGpuSolver.RawGpuResult result = gpuSolver.solveFullEvolutionBatch(dummy, inflows, dummy, stride);

        // --- ASSERT ---
        verify(mockNativeSolver).runBatch(anyLong(), any(), any(), eq(batchSize), eq(INativeManningSolver.MODE_FULL_EVOLUTION), eq(stride));

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