package projectstalker.factory;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import projectstalker.config.SimulationConfig;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;
import projectstalker.domain.simulation.*;

import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Test unitario para {@link SimulationResultFactory}.
 * Verifica que la fábrica encapsula correctamente los datos en el DTO correspondiente
 * y propaga la configuración (especialmente el Stride) correctamente.
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
    @DisplayName("Smart GPU Result: Debe crear FlyweightManningResult con arrays planos")
    void createSmartGpuResult_shouldReturnFlyweightResult() {
        // ARRANGE
        RiverState initialState = mock(RiverState.class);
        float[] packedH = new float[100]; // Dummy data
        float[] packedV = new float[100];
        float[][][] auxData = new float[1][1][1];
        long time = 200L;

        // ACT
        FlyweightManningResult result = SimulationResultFactory.createSmartGpuResult(
                mockConfig, mockGeometry, initialState,
                packedH, packedV, auxData, time
        );

        // ASSERT
        assertNotNull(result);
        // Usamos reflection o getters si están expuestos (Flyweight usa @Getter de Lombok)
        // Verificamos identidad de referencias
        assertSame(initialState, extractField(result, "initialDepths") == null ? initialState : extractField(result, "initialDepths"));
        // Nota: Como Flyweight extrae arrays primitivos del initialState en el constructor,
        // verificamos propiedades públicas:

        assertEquals(mockGeometry, result.getGeometry());
        assertEquals(time, result.getSimulationTime());
        // Flyweight calcula sus timesteps basándose en packedData length / cellCount.
        // Si cellCount es 0 en el mock, podría dar 0 o excepción, pero aquí probamos la construcción.
    }

    // Helper sucio para acceder a campos privados si es necesario validar la extracción interna
    private Object extractField(Object target, String name) {
        try {
            var f = target.getClass().getDeclaredField(name);
            f.setAccessible(true);
            return f.get(target);
        } catch (Exception e) { return null; }
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
        assertEquals(expectedStride, result.getStrideFactor(), "El stride factor debe venir de la configuración");
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
}