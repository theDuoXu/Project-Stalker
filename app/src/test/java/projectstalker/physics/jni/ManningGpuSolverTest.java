package projectstalker.physics.jni;

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
 * Actualizado para validar la arquitectura Zero-Copy.
 * Verifica que el solver transforma los arrays de la geometría en FloatBuffers
 * antes de pasarlos a la capa nativa.
 */
class ManningGpuSolverTest {

    private ManningGpuSolver gpuSolver;
    private INativeManningSolver mockNativeSolver;
    private RiverGeometry mockGeometry;

    private final int CELL_COUNT = 3;
    private final int BATCH_SIZE = 2;

    @BeforeEach
    void setUp() {
        // 1. Mock del Native Solver
        mockNativeSolver = mock(INativeManningSolver.class);

        // 2. SUT
        gpuSolver = new ManningGpuSolver(mockNativeSolver);

        // 3. Mock de Geometría
        // Simulamos que la geometría devuelve arrays float[]
        mockGeometry = mock(RiverGeometry.class);
        when(mockGeometry.getCellCount()).thenReturn(CELL_COUNT);
        when(mockGeometry.getSpatialResolution()).thenReturn(100.0f);

        // Arrays de datos simulados
        float[] width = {10.0f, 10.0f, 10.0f};
        float[] slope = {0.0f, 0.0f, 0.0f};
        float[] manning = {0.03f, 0.03f, 0.03f};
        float[] elevation = {10.0f, 9.5f, 9.0f}; // Necesario para calcular bedSlope

        // Mockeamos los getters (o clones si mantuviste el nombre anterior)
        when(mockGeometry.cloneBottomWidth()).thenReturn(width);
        when(mockGeometry.cloneSideSlope()).thenReturn(slope);
        when(mockGeometry.cloneManningCoefficient()).thenReturn(manning);
        when(mockGeometry.cloneElevationProfile()).thenReturn(elevation);
    }

    @Test
    @DisplayName("solveBatch debe crear DirectBuffers para la geometría y llamar al nativo")
    void solveBatch_shouldPrepareBuffers_andCallNative() {

        // --- ARRANGE ---
        float[] initialGuess = {1.5f, 0.0001f, -0.5f};
        float[][] discharges = {
                {10.0f, 0.0f, 5.0f},
                {20.0f, -5.0f, 15.0f}
        };

        // Respuesta simulada del Nativo
        float[] mockGpuResult = {
                1.1f, 0.1f, 1.2f, 0.2f, 1.3f, 0.3f,
                2.1f, 1.1f, 2.2f, 1.2f, 2.3f, 1.3f
        };

        // CONFIGURACIÓN DEL MOCK (CRÍTICO)
        // Aquí definimos que el método nativo espera recibir FloatBuffer en los últimos argumentos
        when(mockNativeSolver.solveManningGpuBatch(
                any(float[].class), // Initial Guess (Dinámico -> Array)
                any(float[].class), // Discharges (Dinámico -> Array)
                eq(BATCH_SIZE),
                eq(CELL_COUNT),
                any(FloatBuffer.class), // BottomWidth -> BUFFER
                any(FloatBuffer.class), // SideSlope -> BUFFER
                any(FloatBuffer.class), // Manning -> BUFFER
                any(FloatBuffer.class)  // BedSlope -> BUFFER
        )).thenReturn(mockGpuResult);

        // --- ACT ---
        float[][][] result = gpuSolver.solveBatch(initialGuess, discharges, mockGeometry);

        // --- ASSERT ---

        // 1. Verificar que se llamó al método con Buffers
        verify(mockNativeSolver, times(1)).solveManningGpuBatch(
                any(float[].class),
                any(float[].class),
                eq(BATCH_SIZE),
                eq(CELL_COUNT),
                any(FloatBuffer.class), // Verifica que NO se pasaron arrays aquí
                any(FloatBuffer.class),
                any(FloatBuffer.class),
                any(FloatBuffer.class)
        );

        // 2. Verificar resultado
        assertNotNull(result);
        assertEquals(1.1f, result[0][0][0], 1e-6f);
    }

    @Test
    @DisplayName("Caching: Llamadas repetidas con la misma geometría no deben recrear buffers")
    void solveBatch_shouldCacheGeometryBuffers() {
        // --- ARRANGE ---
        float[] guess = {1.0f, 1.0f, 1.0f};
        float[][] discharges = {{10f, 10f, 10f}};
        float[] mockResult = {1f, 1f, 1f, 1f, 1f, 1f};

        when(mockNativeSolver.solveManningGpuBatch(any(), any(), anyInt(), anyInt(), any(), any(), any(), any()))
                .thenReturn(mockResult);

        // --- ACT ---
        // Primera llamada
        gpuSolver.solveBatch(guess, discharges, mockGeometry);

        // Segunda llamada (misma geometría)
        gpuSolver.solveBatch(guess, discharges, mockGeometry);

        // --- ASSERT ---
        // Verificamos que los métodos de acceso a datos de la geometría solo se llamaron UNA VEZ
        // (porque la segunda vez se usó la caché)
        verify(mockGeometry, times(1)).cloneBottomWidth();
        verify(mockGeometry, times(1)).cloneManningCoefficient();
        // verify(mockGeometry, times(1)).cloneElevationProfile(); // Este se llama para calcular pendiente
    }
}