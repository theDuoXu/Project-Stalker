package projectstalker.domain.simulation;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Test unitario para {@link FlyweightManningResult}.
 * <p>
 * Verifica la lógica crítica de "Pure Flyweight": reconstrucción del estado del río
 * fusionando un array compacto (GPU) con un array base (Estado Inicial).
 */
class FlyweightManningResultTest {

    // --- Constantes de Prueba ---
    private final int CELL_COUNT = 10;   // Río de 10 celdas
    private final int ACTIVE_WIDTH = 3;  // GPU solo calcula 3 celdas (Buffer compacto)
    private final int TIMESTEPS = 4;     // Simulamos 4 pasos

    private final float INIT_VAL = 10.0f; // Valor de fondo (Profundidad base)

    @Test
    @DisplayName("Cálculo de Timesteps: Debe derivarse del tamaño del array y activeWidth")
    void constructor_shouldCalculateTimestepsCorrectly() {
        // ARRANGE
        int totalElements = TIMESTEPS * ACTIVE_WIDTH; // 12 elementos
        float[] packedData = new float[totalElements];

        RiverGeometry mockGeo = mock(RiverGeometry.class);
        when(mockGeo.getCellCount()).thenReturn(CELL_COUNT);

        RiverState mockState = createMockState();

        // ACT
        FlyweightManningResult result = FlyweightManningResult.builder()
                .geometry(mockGeo)
                .initialState(mockState)
                .packedDepths(packedData)
                .packedVelocities(packedData)
                .activeWidth(ACTIVE_WIDTH)
                .build();

        // ASSERT
        assertEquals(TIMESTEPS, result.getTimestepCount());
    }

    @Test
    @DisplayName("Fusión Ola (t=0): Debe copiar 1 elemento de GPU y 9 de Fondo")
    void getRawWaterDepthAt_TimeZero_ShouldMergeCorrectly() {
        // ARRANGE
        // GPU Data [Time][Cell in ActiveWidth]
        // t=0: [1.1, 1.2, 1.3] (Solo usaremos 1.1 porque t=0 => waveFront=1)
        float[] packedH = createPackedData();

        FlyweightManningResult flyweight = createSUT(packedH);

        // ACT
        float[] result = flyweight.getRawWaterDepthAt(0);

        // ASSERT
        assertEquals(CELL_COUNT, result.length, "El array debe tener el ancho completo del río");

        // Zona GPU (Índice 0) -> Valor 1.1
        assertEquals(1.1f, result[0], 0.001f, "t=0 c=0 debe venir de GPU");

        // Zona Frontera (Índice 1) -> Aunque GPU tiene dato (1.2), la onda física es t+1=1.
        // Por tanto, el índice 1 aún no ha sido alcanzado por la onda en t=0.
        // Debe ser INIT_VAL.
        // NOTA: Si tu lógica asume que GPU ya calculó el triángulo, t=0 implica c=0.
        assertEquals(INIT_VAL, result[1], 0.001f, "t=0 c=1 debe ser fondo (onda no llegó)");

        // Zona Fondo (Índice 9)
        assertEquals(INIT_VAL, result[9], 0.001f, "t=0 c=9 debe ser fondo");
    }

    @Test
    @DisplayName("Fusión Ola (t=1): Debe copiar 2 elementos de GPU y 8 de Fondo")
    void getRawWaterDepthAt_TimeOne_ShouldAdvanceWave() {
        // ARRANGE
        float[] packedH = createPackedData();
        FlyweightManningResult flyweight = createSUT(packedH);

        // ACT
        float[] result = flyweight.getRawWaterDepthAt(1);

        // ASSERT
        // GPU t=1 values: 2.1, 2.2, 2.3
        // waveFront = min(t+1, activeWidth) = 2.

        assertEquals(2.1f, result[0], 0.001f); // GPU t=1, c=0
        assertEquals(2.2f, result[1], 0.001f); // GPU t=1, c=1

        // c=2 debe ser fondo (waveFront=2, indices 0 y 1 ocupados)
        assertEquals(INIT_VAL, result[2], 0.001f);
    }

    @Test
    @DisplayName("Saturación (t=3): La onda supera el buffer, debe copiar todo el buffer y rellenar resto")
    void getRawWaterDepthAt_TimeSaturated_ShouldClampToActiveWidth() {
        // ARRANGE
        float[] packedH = createPackedData(); // activeWidth = 3
        FlyweightManningResult flyweight = createSUT(packedH);

        // ACT
        // Pedimos t=3. WaveFront físico = 4. Pero ActiveWidth = 3.
        // Debe copiar 3 elementos (índices 0, 1, 2).
        float[] result = flyweight.getRawWaterDepthAt(3);

        // ASSERT
        // GPU t=3 values: 4.1, 4.2, 4.3
        assertEquals(4.1f, result[0], 0.001f);
        assertEquals(4.2f, result[1], 0.001f);
        assertEquals(4.3f, result[2], 0.001f);

        // Index 3 debe ser fondo (se acabó el buffer GPU)
        assertEquals(INIT_VAL, result[3], 0.001f);
    }

    @Test
    @DisplayName("Integridad Full Object: getStateAt debe hidratar auxiliares")
    void getStateAt_ShouldHydrateFullObject() {
        // ARRANGE
        float[] packed = createPackedData();
        // Aux Data: 1 batch, 2 variables (T, pH), 1 valor global
        float[][][] aux = new float[TIMESTEPS][2][1];
        aux[0][0][0] = 25.0f; // Temp at t=0
        aux[0][1][0] = 8.5f;  // pH at t=0

        FlyweightManningResult flyweight = FlyweightManningResult.builder()
                .geometry(mockGeometry())
                .initialState(createMockState())
                .packedDepths(packed)
                .packedVelocities(packed)
                .activeWidth(ACTIVE_WIDTH)
                .auxData(aux)
                .build();

        // ACT
        RiverState state = flyweight.getStateAt(0);

        // ASSERT
        assertNotNull(state);
        // Verificar Física
        assertEquals(1.1f, state.getWaterDepthAt(0));
        assertEquals(INIT_VAL, state.getWaterDepthAt(1));

        // Verificar Química (rellenado con valor global)
        assertEquals(25.0f, state.getTemperatureAt(0));
        assertEquals(25.0f, state.getTemperatureAt(9)); // Todo el río igual
        assertEquals(8.5f, state.getPhAt(5));
    }

    // --- Helpers de Datos ---

    /**
     * Crea un array plano simulando datos GPU.
     * Estructura: 4 pasos de tiempo, 3 celdas de ancho.
     * Valores: (t+1) + (c+1)/10.
     * Ej t=0: [1.1, 1.2, 1.3]
     * Ej t=1: [2.1, 2.2, 2.3]
     */
    private float[] createPackedData() {
        float[] data = new float[TIMESTEPS * ACTIVE_WIDTH];
        for (int t = 0; t < TIMESTEPS; t++) {
            for (int c = 0; c < ACTIVE_WIDTH; c++) {
                int index = t * ACTIVE_WIDTH + c;
                data[index] = (float) (t + 1) + (float) (c + 1) / 10.0f;
            }
        }
        return data;
    }

    private RiverState createMockState() {
        float[] init = new float[CELL_COUNT];
        Arrays.fill(init, INIT_VAL);

        RiverState state = mock(RiverState.class);
        when(state.waterDepth()).thenReturn(init);
        when(state.velocity()).thenReturn(init);
        return state;
    }

    private RiverGeometry mockGeometry() {
        RiverGeometry geo = mock(RiverGeometry.class);
        when(geo.getCellCount()).thenReturn(CELL_COUNT);
        return geo;
    }

    private FlyweightManningResult createSUT(float[] packedData) {
        return FlyweightManningResult.builder()
                .geometry(mockGeometry())
                .initialState(createMockState())
                .packedDepths(packedData)
                .packedVelocities(packedData) // Usamos el mismo para simplificar
                .activeWidth(ACTIVE_WIDTH)
                .build();
    }
}