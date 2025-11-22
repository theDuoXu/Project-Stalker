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
 * Actualizado para soportar la arquitectura de Zero-Copy con FloatBuffers.
 * Verifica que la lógica Java transforme correctamente los arrays a Buffers
 * antes de llamar a JNI.
 */
class ManningGpuSolverTest {

    private ManningGpuSolver gpuSolver;
    private INativeManningSolver mockNativeSolver;
    private RiverGeometry mockGeometry;

    private final int CELL_COUNT = 3;
    private final int BATCH_SIZE = 2;

    @BeforeEach
    void setUp() {
        mockNativeSolver = mock(INativeManningSolver.class);
        gpuSolver = new ManningGpuSolver(mockNativeSolver);

        mockGeometry = mock(RiverGeometry.class);
        when(mockGeometry.getCellCount()).thenReturn(CELL_COUNT);
        when(mockGeometry.getSpatial_resolution()).thenReturn(100.0f);

        // Mockear datos de geometría (Arrays float[])
        // Nota: Asegúrate de que estos métodos coinciden con tu RiverGeometry refactorizada
        when(mockGeometry.getBottomWidth()).thenReturn(new float[]{10.0f, 10.0f, 10.0f});
        when(mockGeometry.getSideSlope()).thenReturn(new float[]{0.0f, 0.0f, 0.0f});
        when(mockGeometry.getManningCoefficient()).thenReturn(new float[]{0.03f, 0.03f, 0.03f});

        // Para bedSlope, el solver calcula la pendiente internamente usando el perfil de elevación
        when(mockGeometry.cloneElevationProfile()).thenReturn(new float[]{10.0f, 9.5f, 9.0f});
    }

    @Test
    @DisplayName("solveBatch debe preparar Buffers, llamar al nativo y desempaquetar")
    void solveBatch_shouldPrepareData_callNative_andUnpackResults() {

        // --- ARRANGE ---
        float[] initialGuess = {1.5f, 0.0001f, -0.5f};
        float[][] discharges = {
                {10.0f, 0.0f, 5.0f},
                {20.0f, -5.0f, 15.0f}
        };

        // Datos esperados (Sanitizados y Expandidos)
        float[] expectedSanitizedGuess = {1.5f, 0.001f, 0.001f, 1.5f, 0.001f, 0.001f};
        float[] expectedFlatDischarges = {10.0f, 0.001f, 5.0f, 20.0f, 0.001f, 15.0f};

        // Respuesta simulada del Nativo (flat array)
        float[] mockGpuResult = {
                1.1f, 0.1f, 1.2f, 0.2f, 1.3f, 0.3f, // Batch 0
                2.1f, 1.1f, 2.2f, 1.2f, 2.3f, 1.3f  // Batch 1
        };

        // CONFIGURACIÓN DEL MOCK (La parte crítica del refactoring)
        // Ahora esperamos FloatBuffer para la geometría, no float[]
        when(mockNativeSolver.solveManningGpuBatch(
                any(float[].class), // Initial Guess (Dinámico)
                any(float[].class), // Discharges (Dinámico)
                anyInt(), anyInt(),
                any(FloatBuffer.class), // BottomWidth (Estático/Buffer)
                any(FloatBuffer.class), // SideSlope (Estático/Buffer)
                any(FloatBuffer.class), // Manning (Estático/Buffer)
                any(FloatBuffer.class)  // BedSlope (Estático/Buffer)
        )).thenReturn(mockGpuResult);

        ArgumentCaptor<float[]> guessCaptor = ArgumentCaptor.forClass(float[].class);
        ArgumentCaptor<float[]> dischargeCaptor = ArgumentCaptor.forClass(float[].class);

        // --- ACT ---
        float[][][] finalResult = gpuSolver.solveBatch(initialGuess, discharges, mockGeometry);

        // --- ASSERT ---

        // 1. Verificar llamada con tipos correctos
        verify(mockNativeSolver, times(1)).solveManningGpuBatch(
                guessCaptor.capture(),
                dischargeCaptor.capture(),
                eq(BATCH_SIZE),
                eq(CELL_COUNT),
                any(FloatBuffer.class), // Verificamos que se pasaron buffers
                any(FloatBuffer.class),
                any(FloatBuffer.class),
                any(FloatBuffer.class)
        );

        // 2. Verificar datos dinámicos
        assertArrayEquals(expectedSanitizedGuess, guessCaptor.getValue(), 1e-6f);
        assertArrayEquals(expectedFlatDischarges, dischargeCaptor.getValue(), 1e-6f);

        // 3. Verificar desempaquetado
        assertArrayEquals(new float[]{2.1f, 2.2f, 2.3f}, finalResult[1][0], 1e-6f);
        assertArrayEquals(new float[]{1.1f, 1.2f, 1.3f}, finalResult[1][1], 1e-6f);
    }

    @Test
    @DisplayName("flattenDischargeProfiles debe aplanar y sanear correctamente")
    void flattenDischargeProfiles_shouldFlattenAndSanitize() {
        float[][] inputDischarges = {{10.0f, 0.0f, 5.0f}, {20.0f, -5.0f, 15.0f}};
        float[] expectedFlatArray = {10.0f, 0.001f, 5.0f, 20.0f, 0.001f, 15.0f};

        float[] flatDischarges = gpuSolver.flattenDischargeProfiles(inputDischarges);

        assertArrayEquals(expectedFlatArray, flatDischarges, 1e-6f);
    }

    @Test
    @DisplayName("sanitizeInitialDepths debe corregir valores no físicos")
    void sanitizeInitialDepths_shouldSanitizeLowValues() {
        float[] initialGuess = {1.5f, 0.0001f, 1e-3f, 0.5f, -0.1f};
        float[] expectedSanitized = {1.5f, 0.001f, 0.001f, 0.5f, 0.001f};

        float[] sanitizedDepths = gpuSolver.sanitizeInitialDepths(initialGuess);

        assertArrayEquals(expectedSanitized, sanitizedDepths, 1e-6f);
    }
}