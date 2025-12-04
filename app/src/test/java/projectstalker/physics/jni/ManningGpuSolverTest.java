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
 * <p>
 * Adaptado para la nueva arquitectura donde los datos viajan via DirectBuffers
 * y la GPU escribe "in-place" en lugar de devolver arrays.
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

        // Dummy data para geometría
        float[] dummyArray = new float[CELL_COUNT];
        // Llenamos algo básico para que no sean todo ceros
        for(int i=0; i<CELL_COUNT; i++) dummyArray[i] = 1.0f;

        when(mockGeometry.getBottomWidth()).thenReturn(dummyArray);
        when(mockGeometry.getSideSlope()).thenReturn(dummyArray);
        when(mockGeometry.getManningCoefficient()).thenReturn(dummyArray);
        when(mockGeometry.cloneElevationProfile()).thenReturn(dummyArray);

        // 3. SUT: Instanciación (Lazy Init - Aún no llama a nada)
        gpuSolver = new ManningGpuSolver(mockNativeSolver, mockGeometry);
    }

    @AfterEach
    void tearDown() {
        if (gpuSolver != null) {
            try {
                gpuSolver.close();
            } catch (Exception e) {
                // Ignorar
            }
        }
    }

    @Test
    @DisplayName("Lazy Init: solveBatch debe inicializar sesión y pasar 6 Buffers Directos")
    void solveBatch_shouldTriggerLazyInit() {
        // --- ARRANGE ---
        float[] inflows = {100f};
        float[] dummy = new float[CELL_COUNT];

        // Mock runBatch para que devuelva éxito (0) sin hacer nada
        when(mockNativeSolver.runBatch(anyLong(), any(), any(), anyInt())).thenReturn(0);

        // --- ACT ---
        gpuSolver.solveBatch(dummy, inflows, dummy);

        // --- ASSERT ---
        ArgumentCaptor<FloatBuffer> bufferCaptor = ArgumentCaptor.forClass(FloatBuffer.class);

        verify(mockNativeSolver, times(1)).initSession(
                bufferCaptor.capture(), // 1. Width
                bufferCaptor.capture(), // 2. Side
                bufferCaptor.capture(), // 3. Manning
                bufferCaptor.capture(), // 4. Bed
                bufferCaptor.capture(), // 5. InitDepth
                bufferCaptor.capture(), // 6. InitQ
                eq(CELL_COUNT)
        );

        assertEquals(6, bufferCaptor.getAllValues().size());
        assertTrue(bufferCaptor.getAllValues().get(0).isDirect(),
                "Los buffers de inicialización deben ser DirectBuffers para evitar copias");
    }

    @Test
    @DisplayName("DMA Flow: solveBatch debe pasar buffers pinned y desempaquetar lo que la 'GPU' escribe")
    void solveBatch_shouldUseDMA_andUnpackSoAResults() {
        // --- ARRANGE ---
        int batchSize = 2;
        float[] newInflows = {100f, 150f};
        float[] dummy = new float[CELL_COUNT];

        // Datos simulados que la "GPU" escribiría en memoria.
        // Estructura SoA: [H0, H1, H2, H3 | V0, V1, V2, V3]
        // Matriz 2x2 -> 4 elementos por variable.
        float[] gpuWrittenData = {
                // Bloque H (4 floats)
                1.1f, 1.0f,  // T0
                1.5f, 1.4f,  // T1
                // Bloque V (4 floats)
                0.5f, 0.4f,  // T0
                0.8f, 0.7f   // T1
        };

        // SIMULACIÓN DE GPU (Side-Effect Mocking):
        // Cuando se llame a runBatch, escribimos manualmente en el outputBuffer.
        doAnswer(invocation -> {
            FloatBuffer outBuf = invocation.getArgument(2); // Argumento #2 es outputBuffer
            int batchArg = invocation.getArgument(3);

            // Verificamos que el buffer tiene capacidad suficiente
            assertTrue(outBuf.capacity() >= gpuWrittenData.length);

            // Escribimos los datos "calculados"
            outBuf.clear();
            outBuf.put(gpuWrittenData);

            return 0; // Return Success (0)
        }).when(mockNativeSolver).runBatch(
                eq(FAKE_SESSION_HANDLE),
                any(FloatBuffer.class), // Input Buffer
                any(FloatBuffer.class), // Output Buffer
                eq(batchSize)
        );

        // --- ACT ---
        float[][][] result = gpuSolver.solveBatch(dummy, newInflows, dummy);

        // --- ASSERT ---

        // 1. Verificar llamada correcta (Buffer in, Buffer out)
        verify(mockNativeSolver).runBatch(
                eq(FAKE_SESSION_HANDLE),
                any(FloatBuffer.class),
                any(FloatBuffer.class),
                eq(batchSize)
        );

        // 2. Verificar Desempaquetado
        // H(t0, x0) = 1.1
        assertEquals(1.1f, result[0][0][0], 1e-6f);
        // V(t0, x0) = 0.5 (Offset 4)
        assertEquals(0.5f, result[0][1][0], 1e-6f);
        // H(t1, x0) = 1.5
        assertEquals(1.5f, result[1][0][0], 1e-6f);
    }

    @Test
    @DisplayName("Sanitización: Los datos escritos en el InputBuffer deben estar limpios")
    void solveBatch_shouldSanitizeInputs_BeforeWritingToBuffer() {
        // --- ARRANGE ---
        float[] dirtyInflows = {-50f, 0f, 100f}; // Inputs sucios
        float[] dummy = new float[CELL_COUNT];
        int batchSize = 3;

        when(mockNativeSolver.runBatch(anyLong(), any(), any(), anyInt())).thenReturn(0);

        // --- ACT ---
        gpuSolver.solveBatch(dummy, dirtyInflows, dummy);

        // --- ASSERT ---
        // Capturamos el buffer que realmente se envió a C++
        ArgumentCaptor<FloatBuffer> inputCaptor = ArgumentCaptor.forClass(FloatBuffer.class);

        verify(mockNativeSolver).runBatch(
                eq(FAKE_SESSION_HANDLE),
                inputCaptor.capture(),
                any(FloatBuffer.class),
                eq(batchSize)
        );

        FloatBuffer sentBuffer = inputCaptor.getValue();

        // Leemos el contenido del buffer (estamos en test, podemos leerlo)
        // Nota: Como es DirectBuffer, leemos usando absolute get o rewind
        float val0 = sentBuffer.get(0);
        float val1 = sentBuffer.get(1);
        float val2 = sentBuffer.get(2);

        assertEquals(0.001f, val0, 1e-6f, "Input negativo debe ser 0.001");
        assertEquals(0.001f, val1, 1e-6f, "Input cero debe ser 0.001");
        assertEquals(100f,   val2, 1e-6f, "Input positivo debe persistir");
    }

    @Test
    @DisplayName("close() debe liberar la sesión nativa")
    void close_shouldDestroySession() {
        // --- ARRANGE ---
        float[] dummy = {100f};
        float[] bigDummy = new float[CELL_COUNT];
        when(mockNativeSolver.runBatch(anyLong(), any(), any(), anyInt())).thenReturn(0);

        // Forzar init
        gpuSolver.solveBatch(bigDummy, dummy, bigDummy);

        // --- ACT ---
        gpuSolver.close();

        // --- ASSERT ---
        verify(mockNativeSolver, times(1)).destroySession(FAKE_SESSION_HANDLE);
    }
}