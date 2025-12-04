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
 * 1. El ciclo de vida RAII (Lazy Init -> Run -> Destroy).
 * 2. La conversión de Geometría y Estado Base a DirectBuffers solo en la primera llamada.
 * 3. El paso de datos "Flyweight" (solo inflows) en solveBatch.
 * 4. El desempaquetado correcto de resultados SoA (Structure of Arrays).
 */
class ManningGpuSolverTest {

    private ManningGpuSolver gpuSolver;
    private INativeManningSolver mockNativeSolver;
    private RiverGeometry mockGeometry;

    private final int CELL_COUNT = 50; // Usamos un río grande para probar el recorte triangular
    private final long FAKE_SESSION_HANDLE = 12345L;

    @BeforeEach
    void setUp() {
        // 1. Mock del Native Solver
        mockNativeSolver = mock(INativeManningSolver.class);

        // Configurar el mock para que devuelva un handle válido al iniciar sesión
        // AHORA ACEPTA 6 BUFFERS (4 Geometría + 2 Estado Inicial)
        when(mockNativeSolver.initSession(any(), any(), any(), any(), any(), any(), eq(CELL_COUNT)))
                .thenReturn(FAKE_SESSION_HANDLE);

        // 2. Mock de Geometría (Necesario para el Constructor del Solver)
        mockGeometry = mock(RiverGeometry.class);
        when(mockGeometry.getCellCount()).thenReturn(CELL_COUNT);
        when(mockGeometry.getSpatialResolution()).thenReturn(10.0f); // dx para calcular pendientes

        // Datos de geometría simulados (tamaño CELL_COUNT)
        float[] width = new float[CELL_COUNT];
        float[] slope = new float[CELL_COUNT];
        float[] manning = new float[CELL_COUNT];
        float[] elevation = new float[CELL_COUNT];

        // Llenamos con dummy data
        for(int i=0; i<CELL_COUNT; i++) {
            width[i] = 10f; slope[i] = 1f; manning[i] = 0.03f; elevation[i] = 10f - i;
        }

        when(mockGeometry.getBottomWidth()).thenReturn(width);
        when(mockGeometry.getSideSlope()).thenReturn(slope);
        when(mockGeometry.getManningCoefficient()).thenReturn(manning);
        when(mockGeometry.cloneElevationProfile()).thenReturn(elevation); // Usado para calcular bedSlope internamente

        // 3. SUT: Instanciación (NO dispara initSession todavía - Lazy)
        gpuSolver = new ManningGpuSolver(mockNativeSolver, mockGeometry);
    }

    @AfterEach
    void tearDown() {
        // Asegurar limpieza si el test no llamó a close explícitamente
        if (gpuSolver != null) {
            try {
                // Solo si se inicializó (handle != 0) se llamará a destroy.
                // Como es lazy, en algunos tests puede no haberse llamado a init.
                // Pero close() maneja eso internamente chequeando el handle.
                gpuSolver.close();
            } catch (Exception e) {
                // Ignorar en teardown
            }
        }
    }

    @Test
    @DisplayName("Lazy Init: solveBatch debe inicializar sesión nativa con 6 Buffers en la primera llamada")
    void solveBatch_shouldTriggerLazyInit() {
        // --- ARRANGE ---
        float[] inflows = {100f};
        float[] depths = new float[CELL_COUNT];
        float[] q = new float[CELL_COUNT];

        // Mock de runBatch para que no falle (retorna array vacío válido para batch 1)
        // Tamaño esperado: 2 * Batch^2 = 2 * 1^2 = 2 floats
        when(mockNativeSolver.runBatch(anyLong(), any())).thenReturn(new float[2]);

        // --- ACT ---
        gpuSolver.solveBatch(depths, inflows, q);

        // --- ASSERT ---
        // Verificamos que se llamó a initSession
        ArgumentCaptor<FloatBuffer> bufferCaptor = ArgumentCaptor.forClass(FloatBuffer.class);

        verify(mockNativeSolver, times(1)).initSession(
                bufferCaptor.capture(), // Width
                bufferCaptor.capture(), // Side
                bufferCaptor.capture(), // Manning
                bufferCaptor.capture(), // Bed
                bufferCaptor.capture(), // InitDepth (Nuevo)
                bufferCaptor.capture(), // InitQ (Nuevo)
                eq(CELL_COUNT)
        );

        // Verificamos que se pasaron 6 buffers distintos
        assertEquals(6, bufferCaptor.getAllValues().size());
        assertTrue(bufferCaptor.getAllValues().get(0).isDirect(), "Los buffers deben ser directos para Zero-Copy");
    }

    @Test
    @DisplayName("solveBatch debe llamar a runBatch solo con inflows y desempaquetar SoA triangular")
    void solveBatch_shouldCallNativeRun_andUnpackSoAResults() {
        // --- ARRANGE ---
        int batchSize = 2; // Matriz 2x2

        // Inputs comprimidos (1D)
        float[] newInflows = {100f, 150f}; // [BatchSize]
        float[] initialDepths = new float[CELL_COUNT];
        float[] initialQ = new float[CELL_COUNT];

        // Output simulado de la GPU (SoA + Triangular)
        // Tamaño total = BatchSize * BatchSize * 2 = 2 * 2 * 2 = 8 floats.
        // Estructura: [ Bloque H (4 floats) | Bloque V (4 floats) ]
        // Bloque H (flattened 2x2): [H(t0,x0), H(t0,x1), H(t1,x0), H(t1,x1)]

        float[] gpuRawOutput = {
                // --- BLOQUE H ---
                1.1f, 1.0f,  // Step 0: H en celda 0, celda 1 (Nota: celda 1 no activa físicamente, pero la matriz es cuadrada)
                1.5f, 1.4f,  // Step 1: H en celda 0, celda 1

                // --- BLOQUE V ---
                0.5f, 0.4f,  // Step 0: V en celda 0, celda 1
                0.8f, 0.7f   // Step 1: V en celda 0, celda 1
        };

        when(mockNativeSolver.runBatch(
                eq(FAKE_SESSION_HANDLE),
                any(float[].class)) // Solo inflows
        ).thenReturn(gpuRawOutput);

        // --- ACT ---
        // Primera llamada (dispara init también)
        float[][][] result = gpuSolver.solveBatch(initialDepths, newInflows, initialQ);

        // --- ASSERT ---

        // 1. Verificar llamada al nativo (Firma ligera)
        verify(mockNativeSolver).runBatch(
                eq(FAKE_SESSION_HANDLE),
                eq(newInflows)    // Verifica paso directo de arrays
        );

        // 2. Verificar Desempaquetado [Batch][Var][Batch]
        // Dimensiones esperadas: [2][2][2] (No CellCount 50)
        assertEquals(batchSize, result.length);
        assertEquals(batchSize, result[0][0].length);

        // Verificamos lectura SoA correcta
        // H(t0, x0) = 1.1 (Índice 0)
        assertEquals(1.1f, result[0][0][0], 1e-6f, "Step 0, Cell 0, Depth incorrecta");

        // V(t0, x0) = 0.5 (Índice 4 - offset de bloque)
        assertEquals(0.5f, result[0][1][0], 1e-6f, "Step 0, Cell 0, Velocity incorrecta");

        // H(t1, x0) = 1.5 (Índice 2)
        assertEquals(1.5f, result[1][0][0], 1e-6f, "Step 1, Cell 0, Depth incorrecta");
    }

    @Test
    @DisplayName("Sanitización: Inputs negativos deben ser corregidos antes de llamar a GPU")
    void solveBatch_shouldSanitizeInputs() {
        // --- ARRANGE ---
        float[] dirtyInflows = {-50f, 0f, 100f};
        float[] depths = new float[CELL_COUNT];
        float[] q = new float[CELL_COUNT];

        // Mock respuesta vacía válida (3*3*2 = 18 floats)
        float[] dummyResult = new float[3 * 3 * 2];
        when(mockNativeSolver.runBatch(anyLong(), any())).thenReturn(dummyResult);

        // --- ACT ---
        gpuSolver.solveBatch(depths, dirtyInflows, q);

        // --- ASSERT ---
        ArgumentCaptor<float[]> inflowCaptor = ArgumentCaptor.forClass(float[].class);

        // Solo verificamos runBatch, initSession recibe depths sanitizados pero es otro método
        verify(mockNativeSolver).runBatch(
                eq(FAKE_SESSION_HANDLE),
                inflowCaptor.capture()
        );

        // Verificar Inflows
        float[] sentInflows = inflowCaptor.getValue();
        assertEquals(0.001f, sentInflows[0], 1e-6f, "Negativo debe ser 0.001");
        assertEquals(0.001f, sentInflows[1], 1e-6f, "Cero debe ser 0.001");
        assertEquals(100f, sentInflows[2], 1e-6f, "Positivo debe quedar igual");
    }

    @Test
    @DisplayName("close() debe llamar a destroySession y anular el handle")
    void close_shouldDestroySession() {
        // --- ARRANGE ---
        // Forzamos inicialización para tener un handle válido que cerrar
        float[] dummy = {1f};
        float[] dummyBig = new float[CELL_COUNT];
        when(mockNativeSolver.runBatch(anyLong(), any())).thenReturn(new float[2]);

        gpuSolver.solveBatch(dummyBig, dummy, dummyBig); // Init

        // --- ACT ---
        gpuSolver.close();

        // --- ASSERT ---
        verify(mockNativeSolver, times(1)).destroySession(FAKE_SESSION_HANDLE);

        // Intentar usarlo después de cerrar debe lanzar excepción Y RE-INICIALIZAR (o fallar, según diseño)
        // En tu diseño actual, checkea (sessionHandle == 0) -> initializeSession.
        // Así que si llamas a solveBatch de nuevo, volverá a inicializar.
        // Si queremos que falle, tendríamos que cambiar el diseño.
        // Pero RAII permite revivir.

        // Verificamos comportamiento: Llama a initSession de nuevo
        gpuSolver.solveBatch(dummyBig, dummy, dummyBig);
        verify(mockNativeSolver, times(2)).initSession(any(), any(), any(), any(), any(), any(), anyInt());
    }
}