package projectstalker.domain.simulation;

import lombok.Builder;
import lombok.Getter;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;

import java.util.Arrays;

/**
 * Implementación "Flyweight" (Lazy) de los resultados de la simulación.
 */
public class FlyweightManningResult implements ISimulationResult {

    @Getter
    private final RiverGeometry geometry;

    @Getter
    private final long simulationTime;

    // --- ESTADO INTRÍNSECO (Compartido / Base) ---
    private final float[] initialDepths;
    private final float[] initialVelocities;

    // --- ESTADO EXTRÍNSECO (Deltas de GPU) ---
    private final float[][][] gpuPackedData;

    // Datos auxiliares
    private final float[][][] auxData;

    /**
     * Constructor personalizado.
     * Ponemos @Builder AQUÍ para que Lombok use este constructor.
     * Esto permite pasar 'RiverState' al builder y que se desempaquete internamente.
     */
    @Builder
    public FlyweightManningResult(RiverGeometry geometry,
                                  long simulationTime,
                                  RiverState initialState, // El builder aceptará RiverState
                                  float[][][] gpuPackedData,
                                  float[][][] auxData) {
        this.geometry = geometry;
        this.simulationTime = simulationTime;

        // Extracción defensiva inicial (Flyweight Base)
        this.initialDepths = initialState.waterDepth();
        this.initialVelocities = initialState.velocity();

        this.gpuPackedData = gpuPackedData;
        this.auxData = auxData;
    }

    @Override
    public int getTimestepCount() {
        return gpuPackedData.length;
    }

    /**
     * Reconstruye el estado del río en el tiempo 't' bajo demanda.
     */
    @Override
    public RiverState getStateAt(int t) {
        int cellCount = geometry.getCellCount();

        float[] h = new float[cellCount];
        float[] v = new float[cellCount];

        // 1. ZONA NUEVA (GPU Data)
        float[] gpuH = gpuPackedData[t][0];
        float[] gpuV = gpuPackedData[t][1];

        int newDataLimit = Math.min(t + 1, gpuH.length);
        int copyLength = Math.min(newDataLimit, cellCount);

        System.arraycopy(gpuH, 0, h, 0, copyLength);
        System.arraycopy(gpuV, 0, v, 0, copyLength);

        // 2. ZONA ANTIGUA (Estado Inicial Desplazado)
        int remainingCells = cellCount - copyLength;

        if (remainingCells > 0) {
            // Desplazamiento: initial[0] pasa a ser h[copyLength]
            System.arraycopy(initialDepths, 0, h, copyLength, remainingCells);
            System.arraycopy(initialVelocities, 0, v, copyLength, remainingCells);
        }

        // 3. CONSTRUCCIÓN DEL ESTADO
        float t_val = (auxData != null && t < auxData.length) ? auxData[t][0][0] : 0f;
        float ph_val = (auxData != null && t < auxData.length) ? auxData[t][1][0] : 0f;

        float[] tempArr = new float[cellCount];
        float[] phArr = new float[cellCount];
        Arrays.fill(tempArr, t_val);
        Arrays.fill(phArr, ph_val);

        return RiverState.builder()
                .waterDepth(h)
                .velocity(v)
                .temperature(tempArr)
                .ph(phArr)
                .contaminantConcentration(new float[cellCount])
                .build();
    }
}