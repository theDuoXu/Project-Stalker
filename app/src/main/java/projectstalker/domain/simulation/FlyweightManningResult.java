package projectstalker.domain.simulation;

import lombok.Builder;
import lombok.Getter;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;

import java.util.Arrays;

/**
 * Implementación "Flyweight" (Lazy) de los resultados de la simulación.
 * Esta compresión solo funciona con ríos con caudal equilibrado (en estado de reposo).
 * <p>
 * Implementa {@link IManningResult} para permitir acceso rápido a los arrays crudos.
 */
public class FlyweightManningResult implements IManningResult {

    @Getter
    private final RiverGeometry geometry;

    @Getter
    private final long simulationTime;

    // --- ESTADO INTRÍNSECO (Compartido / Base) ---
    private final float[] initialDepths;
    private final float[] initialVelocities;

    // --- ESTADO EXTRÍNSECO (Deltas de GPU) ---
    // [Tiempo][Variable (0=H, 1=V)][Celda]
    private final float[][][] gpuPackedData;

    // Datos auxiliares
    private final float[][][] auxData;

    @Builder
    public FlyweightManningResult(RiverGeometry geometry,
                                  long simulationTime,
                                  RiverState initialState,
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

    // --- IMPLEMENTACIÓN IManningResult (Fast Path) ---

    @Override
    public float[] getRawWaterDepthAt(int t) {
        // Variable 0 = Profundidad en gpuPackedData
        return buildCompositeArray(t, 0, initialDepths);
    }

    @Override
    public float[] getRawVelocityAt(int t) {
        // Variable 1 = Velocidad en gpuPackedData
        return buildCompositeArray(t, 1, initialVelocities);
    }

    /**
     * Helper centralizado para construir el array híbrido.
     * Combina la ola dinámica (GPU) con el estado base (Initial).
     */
    private float[] buildCompositeArray(int t, int gpuVarIndex, float[] initialBackground) {
        int cellCount = geometry.getCellCount();
        float[] result = new float[cellCount];

        // 1. ZONA NUEVA (GPU Data)
        float[] gpuData = gpuPackedData[t][gpuVarIndex];

        // La GPU nos devuelve datos válidos hasta el frente de onda
        // (Asumiendo propagación CFL <= 1 por paso de tiempo)
        int newDataLimit = Math.min(t + 1, gpuData.length);
        int copyLength = Math.min(newDataLimit, cellCount);

        System.arraycopy(gpuData, 0, result, 0, copyLength);

        // 2. ZONA ANTIGUA (Estado Base Estacionario)
        int remainingCells = cellCount - copyLength;

        if (remainingCells > 0) {
            // Preservamos el estado inicial en la zona que la ola aún no ha alcanzado.
            // Copiamos desde copyLength hacia copyLength para mantener coherencia espacial.
            System.arraycopy(initialBackground, copyLength, result, copyLength, remainingCells);
        }

        return result;
    }

    // --- IMPLEMENTACIÓN ISimulationResult (Full Object) ---

    @Override
    public RiverState getStateAt(int t) {
        // Reutilizamos la lógica del Fast Path para obtener H y V
        float[] h = getRawWaterDepthAt(t);
        float[] v = getRawVelocityAt(t);

        // 3. CONSTRUCCIÓN DE AUXILIARES (Química)
        int cellCount = geometry.getCellCount();

        float t_val = (auxData != null && t < auxData.length) ? auxData[t][0][0] : 0f;
        float ph_val = (auxData != null && t < auxData.length) ? auxData[t][1][0] : 0f;

        float[] tempArr = new float[cellCount];
        float[] phArr = new float[cellCount];

        // Rellenado eficiente (JVM intrinsic)
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