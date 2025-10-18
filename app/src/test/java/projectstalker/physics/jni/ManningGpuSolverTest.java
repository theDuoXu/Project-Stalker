package projectstalker.physics.jni;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import projectstalker.domain.river.RiverGeometry;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

/**
 * Test unitario para ManningGpuSolver.
 * Este test utiliza Inyección de Dependencias para mockear la capa nativa
 * (INativeManningSolver) y probar la lógica de orquestación de la clase.
 */
class ManningGpuSolverTest {

    // --- Mocks y SUT (Subject Under Test) ---
    private ManningGpuSolver gpuSolver; // La clase que estamos probando
    private INativeManningSolver mockNativeSolver; // El mock de la capa nativa
    private RiverGeometry mockGeometry;

    private final int CELL_COUNT = 3;
    private final int BATCH_SIZE = 2;

    @BeforeEach
    void setUp() {
        // 1. Crear el mock de la dependencia nativa
        mockNativeSolver = mock(INativeManningSolver.class);

        // 2. Instanciar la clase bajo prueba (SUT) inyectando el mock
        gpuSolver = new ManningGpuSolver(mockNativeSolver);

        // 3. Configurar mocks de geometría (necesarios para createGpuGeometry)
        mockGeometry = mock(RiverGeometry.class);
        when(mockGeometry.getCellCount()).thenReturn(CELL_COUNT);
        when(mockGeometry.cloneBottomWidth()).thenReturn(new double[]{10.0, 10.0, 10.0});
        when(mockGeometry.cloneSideSlope()).thenReturn(new double[]{0.0, 0.0, 0.0});
        when(mockGeometry.cloneManningCoefficient()).thenReturn(new double[]{0.03, 0.03, 0.03});
        when(mockGeometry.cloneElevationProfile()).thenReturn(new double[]{10.0, 9.5, 9.0});
        when(mockGeometry.getDx()).thenReturn(100.0);
    }

    @Test
    @DisplayName("solveBatch debe sanear, aplanar, llamar al nativo y desempaquetar la respuesta")
    void solveBatch_shouldPrepareData_callNative_andUnpackResults() {

        // --- ARRANGE ---

        // 1. Datos de ENTRADA para el método solveBatch
        double[] initialGuess = {1.5, 0.0001, -0.5}; // Necesita saneamiento
        double[][] discharges = {
                {10.0, 0.0, 5.0},    // t=0, necesita saneamiento
                {20.0, -5.0, 15.0}   // t=1, necesita saneamiento
        };

        // 2. Datos PROCESADOS que esperamos que se envíen al método nativo
        float[] expectedSanitizedGuess = {1.5f, 0.001f, 0.001f};
        float[] expectedFlatDischarges = {10.0f, 0.001f, 5.0f, 20.0f, 0.001f, 15.0f};

        // 3. Datos de VUELTA (simulados) que devuelve el mock nativo
        // Formato: [D_t0_c0, V_t0_c0, D_t0_c1, V_t0_c1, ..., D_t1_c0, V_t1_c0, ...]
        float[] mockGpuResult = {
                // Batch 0
                1.1f, 0.1f,  // Celda 0
                1.2f, 0.2f,  // Celda 1
                1.3f, 0.3f,  // Celda 2
                // Batch 1
                2.1f, 1.1f,  // Celda 0
                2.2f, 1.2f,  // Celda 1
                2.3f, 1.3f   // Celda 2
        };

        // 4. Configurar el mock para que devuelva los datos simulados
        when(mockNativeSolver.solveManningGpuBatch(
                any(float[].class), any(float[].class),
                anyInt(), anyInt(),
                any(float[].class), any(float[].class), any(float[].class), any(float[].class)
        )).thenReturn(mockGpuResult);

        // 5. Preparar ArgumentCaptors para verificar los datos enviados al mock
        ArgumentCaptor<float[]> guessCaptor = ArgumentCaptor.forClass(float[].class);
        ArgumentCaptor<float[]> dischargeCaptor = ArgumentCaptor.forClass(float[].class);

        // --- ACT ---
        double[][][] finalResult = gpuSolver.solveBatch(initialGuess, discharges, mockGeometry);

        // --- ASSERT ---

        // 1. Verificar que el método nativo fue llamado UNA VEZ
        verify(mockNativeSolver, times(1)).solveManningGpuBatch(
                guessCaptor.capture(),
                dischargeCaptor.capture(),
                eq(BATCH_SIZE),
                eq(CELL_COUNT),
                any(float[].class), any(float[].class), any(float[].class), any(float[].class)
        );

        // 2. Verificar que los datos enviados al nativo fueron saneados y aplanados
        assertArrayEquals(expectedSanitizedGuess, guessCaptor.getValue(), 1e-6f, "La profundidad inicial no fue saneada correctamente.");
        assertArrayEquals(expectedFlatDischarges, dischargeCaptor.getValue(), 1e-6f, "Los caudales no fueron aplanados o saneados correctamente.");

        // 3. Verificar que los datos devueltos (finalResult) fueron desempaquetados
        double[] expectedDepthsBatch1 = {2.1, 2.2, 2.3};
        double[] expectedVelsBatch1 = {1.1, 1.2, 1.3};

        assertArrayEquals(expectedDepthsBatch1, finalResult[1][0], 1e-6, "Las profundidades del batch 1 no se desempaquetaron correctamente.");
        assertArrayEquals(expectedVelsBatch1, finalResult[1][1], 1e-6, "Las velocidades del batch 1 no se desempaquetaron correctamente.");
    }

    // --- Tests para los métodos de ayuda (ahora que no son privados) ---
    // (Estos son los tests que escribiste, ahora sin reflexión)

    @Test
    @DisplayName("El aplanamiento debe sanear valores negativos/cero y aplanar el array 2D a 1D")
    void flattenDischargeProfiles_shouldFlattenAndSanitize() {
        double[][] inputDischarges = {{10.0, 0.0, 5.0}, {20.0, -5.0, 15.0}};
        float[] expectedFlatArray = {10.0f, 0.001f, 5.0f, 20.0f, 0.001f, 15.0f};

        float[] flatDischarges = gpuSolver.flattenDischargeProfiles(inputDischarges);

        assertArrayEquals(expectedFlatArray, flatDischarges, 1e-6f);
    }

    @Test
    @DisplayName("La sanitización de profundidades iniciales debe reemplazar valores bajos/cero por 0.001f")
    void sanitizeInitialDepths_shouldSanitizeLowValues() {
        double[] initialGuess = {1.5, 0.0001, 1e-3, 0.5, -0.1};
        float[] expectedSanitized = {1.5f, 0.001f, 0.001f, 0.5f, 0.001f};

        float[] sanitizedDepths = gpuSolver.sanitizeInitialDepths(initialGuess);

        assertArrayEquals(expectedSanitized, sanitizedDepths, 1e-6f);
    }
}