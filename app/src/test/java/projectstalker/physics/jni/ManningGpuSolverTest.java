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
 * Test unitario para ManningGpuSolver (Versión Stateful Refactorizada).
 * <p>
 * Verifica:
 * 1. El ciclo de vida RAII (Init -> Run -> Destroy).
 * 2. La conversión de Geometría a DirectBuffers solo en la inicialización.
 * 3. El paso de datos "Smart Fetch" (arrays primitivos comprimidos) en solveBatch.
 * 4. El desempaquetado correcto de resultados planos.
 */
class ManningGpuSolverTest {

    private ManningGpuSolver gpuSolver;
    private INativeManningSolver mockNativeSolver;
    private RiverGeometry mockGeometry;

    private final int CELL_COUNT = 3;
    private final long FAKE_SESSION_HANDLE = 12345L;

    @BeforeEach
    void setUp() {
        // 1. Mock del Native Solver
        mockNativeSolver = mock(INativeManningSolver.class);

        // Configurar el mock para que devuelva un handle válido al iniciar sesión
        when(mockNativeSolver.initSession(any(), any(), any(), any(), eq(CELL_COUNT)))
                .thenReturn(FAKE_SESSION_HANDLE);

        // 2. Mock de Geometría (Necesario para el Constructor del Solver)
        mockGeometry = mock(RiverGeometry.class);
        when(mockGeometry.getCellCount()).thenReturn(CELL_COUNT);
        when(mockGeometry.getSpatialResolution()).thenReturn(10.0f); // dx para calcular pendientes

        // Datos de geometría simulados
        float[] width = {10f, 10f, 10f};
        float[] slope = {1f, 1f, 1f};
        float[] manning = {0.03f, 0.03f, 0.03f};
        float[] elevation = {10f, 9f, 8f}; // Generará pendiente positiva

        when(mockGeometry.getBottomWidth()).thenReturn(width);
        when(mockGeometry.getSideSlope()).thenReturn(slope);
        when(mockGeometry.getManningCoefficient()).thenReturn(manning);
        when(mockGeometry.cloneElevationProfile()).thenReturn(elevation); // Usado para calcular bedSlope internamente

        // 3. SUT: Instanciación (Esto dispara initSession inmediatamente)
        gpuSolver = new ManningGpuSolver(mockNativeSolver, mockGeometry);
    }

    @AfterEach
    void tearDown() {
        // Asegurar limpieza si el test no llamó a close explícitamente
        if (gpuSolver != null) {
            try {
                gpuSolver.close();
            } catch (Exception e) {
                // Ignorar en teardown
            }
        }
    }

    @Test
    @DisplayName("Constructor debe inicializar sesión nativa con Buffers Directos")
    void constructor_shouldInitNativeSession() {
        // El constructor ya se ejecutó en setUp. Verificamos sus efectos secundarios.

        // Capturamos los argumentos para verificar que se pasaron Buffers
        ArgumentCaptor<FloatBuffer> bufferCaptor = ArgumentCaptor.forClass(FloatBuffer.class);

        verify(mockNativeSolver, times(1)).initSession(
                bufferCaptor.capture(), // Width
                bufferCaptor.capture(), // Side
                bufferCaptor.capture(), // Manning
                bufferCaptor.capture(), // Bed
                eq(CELL_COUNT)
        );

        // Verificamos que se pasaron 4 buffers distintos
        assertEquals(4, bufferCaptor.getAllValues().size());
        assertTrue(bufferCaptor.getAllValues().get(0).isDirect(), "Los buffers deben ser directos para Zero-Copy");
    }

    @Test
    @DisplayName("solveBatch debe llamar a runBatch con arrays primitivos y handle correcto")
    void solveBatch_shouldCallNativeRun_andUnpackResults() {
        // --- ARRANGE ---
        int batchSize = 2;

        // Inputs comprimidos (1D)
        float[] newInflows = {100f, 150f}; // [BatchSize]
        float[] initialDepths = {1.0f, 1.0f, 1.0f}; // [CellCount]
        float[] initialQ = {50f, 50f, 50f}; // [CellCount]

        // Output simulado de la GPU (Plano: [Step][Cell][Var])
        // Layout esperado por unpack: [Step0_C0_H, Step0_C0_V, Step0_C1_H...]
        float[] gpuRawOutput = {
                // Step 0 (Input 100)
                1.1f, 0.5f,  // Cell 0: H=1.1, V=0.5
                1.0f, 0.4f,  // Cell 1
                0.9f, 0.3f,  // Cell 2

                // Step 1 (Input 150)
                1.5f, 0.8f,  // Cell 0: H=1.5, V=0.8
                1.4f, 0.7f,  // Cell 1
                1.3f, 0.6f   // Cell 2
        };

        when(mockNativeSolver.runBatch(
                eq(FAKE_SESSION_HANDLE),
                any(float[].class),
                any(float[].class),
                any(float[].class))
        ).thenReturn(gpuRawOutput);

        // --- ACT ---
        float[][][] result = gpuSolver.solveBatch(initialDepths, newInflows, initialQ);

        // --- ASSERT ---

        // 1. Verificar llamada al nativo
        verify(mockNativeSolver).runBatch(
                eq(FAKE_SESSION_HANDLE),
                eq(newInflows),    // Verifica paso directo de arrays (Pinning)
                any(float[].class), // Depths sanitizados (puede ser copia, usamos any)
                eq(initialQ)
        );

        // 2. Verificar Desempaquetado [Batch][Var][Cell]
        assertEquals(batchSize, result.length);

        // Step 0, Var 0 (Depth), Cell 0 -> Esperado 1.1
        assertEquals(1.1f, result[0][0][0], 1e-6f, "Step 0, Cell 0, Depth incorrecta");
        // Step 0, Var 1 (Vel), Cell 0 -> Esperado 0.5
        assertEquals(0.5f, result[0][1][0], 1e-6f, "Step 0, Cell 0, Velocity incorrecta");

        // Step 1, Var 0 (Depth), Cell 0 -> Esperado 1.5
        assertEquals(1.5f, result[1][0][0], 1e-6f, "Step 1, Cell 0, Depth incorrecta");
    }

    @Test
    @DisplayName("Sanitización: Inputs negativos deben ser corregidos antes de llamar a GPU")
    void solveBatch_shouldSanitizeInputs() {
        // --- ARRANGE ---
        float[] dirtyInflows = {-50f, 0f, 100f};
        float[] dirtyDepths = {0f, -1f, 1f};
        float[] q = {1f, 1f, 1f};

        // Mock respuesta vacía válida para no fallar en unpack
        float[] dummyResult = new float[3 * CELL_COUNT * 2];
        when(mockNativeSolver.runBatch(anyLong(), any(), any(), any())).thenReturn(dummyResult);

        // --- ACT ---
        gpuSolver.solveBatch(dirtyDepths, dirtyInflows, q);

        // --- ASSERT ---
        ArgumentCaptor<float[]> inflowCaptor = ArgumentCaptor.forClass(float[].class);
        ArgumentCaptor<float[]> depthCaptor = ArgumentCaptor.forClass(float[].class);

        verify(mockNativeSolver).runBatch(
                eq(FAKE_SESSION_HANDLE),
                inflowCaptor.capture(),
                depthCaptor.capture(),
                eq(q)
        );

        // Verificar Inflows
        float[] sentInflows = inflowCaptor.getValue();
        assertEquals(0.001f, sentInflows[0], 1e-6f, "Negativo debe ser 0.001");
        assertEquals(0.001f, sentInflows[1], 1e-6f, "Cero debe ser 0.001");
        assertEquals(100f, sentInflows[2], 1e-6f, "Positivo debe quedar igual");

        // Verificar Depths
        float[] sentDepths = depthCaptor.getValue();
        assertEquals(0.001f, sentDepths[0], 1e-6f);
    }

    @Test
    @DisplayName("close() debe llamar a destroySession y anular el handle")
    void close_shouldDestroySession() {
        // --- ACT ---
        gpuSolver.close();

        // --- ASSERT ---
        verify(mockNativeSolver, times(1)).destroySession(FAKE_SESSION_HANDLE);

        // Intentar usarlo después de cerrar debe lanzar excepción
        float[] dummy = {1f};
        assertThrows(IllegalStateException.class, () ->
                gpuSolver.solveBatch(dummy, dummy, dummy)
        );
    }
}