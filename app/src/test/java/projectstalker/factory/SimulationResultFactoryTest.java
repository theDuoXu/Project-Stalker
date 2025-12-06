package projectstalker.factory;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import projectstalker.config.SimulationConfig;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;
import projectstalker.domain.simulation.*;

import java.lang.reflect.Field;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Test unitario para {@link SimulationResultFactory}.
 * <p>
 * Verifica que la fábrica encapsula correctamente los datos en el DTO correspondiente,
 * propaga la configuración (Stride, ActiveWidth) y conecta los arrays de datos crudos.
 */
class SimulationResultFactoryTest {

    private final RiverGeometry mockGeometry = mock(RiverGeometry.class);
    private final SimulationConfig mockConfig = mock(SimulationConfig.class);

    @Test
    @DisplayName("CPU Result: Debe crear DenseManningResult con la lista de estados")
    void createCpuResult_shouldReturnDenseResult() {
        // ARRANGE
        List<RiverState> states = Collections.singletonList(mock(RiverState.class));
        long time = 100L;

        // ACT
        DenseManningResult result = SimulationResultFactory.createCpuResult(mockGeometry, states, time);

        // ASSERT
        assertNotNull(result);
        assertEquals(mockGeometry, result.getGeometry());
        assertEquals(time, result.getSimulationTime());
        assertEquals(states, result.getStates());
        assertEquals(1, result.getTimestepCount());
    }

    @Test
    @DisplayName("Smart GPU Result: Debe crear FlyweightManningResult extrayendo arrays y propagando activeWidth")
    void createSmartGpuResult_shouldReturnFlyweightResult() {
        // ARRANGE
        RiverState initialState = mock(RiverState.class);

        // Arrays dummy que simulan el estado interno del río
        float[] initialH = new float[10];
        float[] initialV = new float[10];

        // Configuramos el mock para devolver estos arrays específicos
        when(initialState.waterDepth()).thenReturn(initialH);
        when(initialState.velocity()).thenReturn(initialV);

        float[] packedH = new float[100];
        float[] packedV = new float[100];
        float[][][] auxData = new float[1][1][1];

        int activeWidth = 50; // Ancho del triángulo activo
        long time = 200L;

        // ACT
        // Llamada actualizada con el nuevo parámetro activeWidth
        FlyweightManningResult result = SimulationResultFactory.createSmartGpuResult(
                mockConfig, mockGeometry, initialState,
                packedH, packedV, auxData,
                activeWidth,
                time
        );

        // ASSERT
        assertNotNull(result);
        assertEquals(mockGeometry, result.getGeometry());
        assertEquals(time, result.getSimulationTime());

        // Verificación Interna (Reflexión):
        // 1. Comprobar que extrajo los arrays del estado inicial
        assertSame(initialH, extractField(result, "initialDepths"));
        assertSame(initialV, extractField(result, "initialVelocities"));

        // 2. Comprobar que guardó los arrays packed de la GPU
        assertSame(packedH, extractField(result, "packedDepths"));
        assertSame(packedV, extractField(result, "packedVelocities"));

        // 3. Comprobar que guardó el activeWidth
        assertEquals(activeWidth, extractField(result, "activeWidth"));
    }

    @Test
    @DisplayName("Full Evolution GPU: Debe crear StridedManningResult y propagar el Stride")
    void createStridedGpuResult_shouldReturnStridedResult_withCorrectStride() {
        // ARRANGE
        int expectedStride = 5;
        when(mockConfig.getGpuFullEvolutionStride()).thenReturn(expectedStride);

        float[] packedH = new float[1000];
        float[] packedV = new float[1000];
        int logicSteps = 500;
        long time = 300L;

        // ACT
        StridedManningResult result = SimulationResultFactory.createStridedGpuResult(
                mockConfig, mockGeometry,
                packedH, packedV,
                logicSteps, time
        );

        // ASSERT
        assertNotNull(result);
        assertEquals(expectedStride, result.getStrideFactor(), "El stride debe venir de la configuración");
        assertEquals(logicSteps, result.getLogicTimestepCount());
        assertSame(packedH, result.getPackedDepths());
        assertSame(packedV, result.getPackedVelocities());
    }

    @Test
    @DisplayName("Chunked GPU: Debe crear ChunkedManningResult con lista de arrays")
    void createChunkedGpuResult_shouldReturnChunkedResult() {
        // ARRANGE
        int expectedStride = 10;
        when(mockConfig.getGpuFullEvolutionStride()).thenReturn(expectedStride);

        List<float[]> dChunks = Collections.singletonList(new float[100]);
        List<float[]> vChunks = Collections.singletonList(new float[100]);
        int stepsPerChunk = 50;
        int logicSteps = 500;
        long time = 400L;

        // ACT
        ChunkedManningResult result = SimulationResultFactory.createChunkedGpuResult(
                mockConfig, mockGeometry,
                dChunks, vChunks,
                stepsPerChunk, logicSteps, time
        );

        // ASSERT
        assertNotNull(result);
        assertEquals(expectedStride, result.getStrideFactor());
        assertSame(dChunks, result.getDepthChunks());
        assertSame(vChunks, result.getVelocityChunks());
        assertEquals(stepsPerChunk, result.getStepsPerChunk());
    }

    // --- Helper de Reflexión Robusto ---
    private Object extractField(Object target, String fieldName) {
        try {
            Field field = target.getClass().getDeclaredField(fieldName);
            field.setAccessible(true);
            return field.get(target);
        } catch (Exception e) {
            throw new RuntimeException("Error en test: No se pudo extraer el campo '" + fieldName + "' de " + target.getClass().getSimpleName(), e);
        }
    }
}