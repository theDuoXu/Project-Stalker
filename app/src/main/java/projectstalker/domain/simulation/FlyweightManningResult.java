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
 * REFACTORIZADO: Ahora utiliza arrays planos (Packed) al igual que las implementaciones Strided,
 * evitando la sobrecarga de arrays 3D y conversiones innecesarias.
 */
public class FlyweightManningResult implements IManningResult {

    @Getter
    private final RiverGeometry geometry;

    @Getter
    private final long simulationTime;

    // --- ESTADO INTRÍNSECO (Compartido / Base) ---
    private final float[] initialDepths;
    private final float[] initialVelocities;

    // --- ESTADO EXTRÍNSECO (Deltas de GPU - PLANOS) ---
    // Layout: [t0_c0...t0_cn | t1_c0...t1_cn]
    private final float[] packedDepths;
    private final float[] packedVelocities;

    private final int timestepCount;

    // Datos auxiliares (Mantenemos estructura original para compatibilidad con módulos químicos CPU)
    private final float[][][] auxData;

    @Builder
    public FlyweightManningResult(RiverGeometry geometry,
                                  long simulationTime,
                                  RiverState initialState,
                                  float[] packedDepths,
                                  float[] packedVelocities,
                                  float[][][] auxData) {
        this.geometry = geometry;
        this.simulationTime = simulationTime;

        // Extracción defensiva inicial (Flyweight Base)
        this.initialDepths = initialState.waterDepth();
        this.initialVelocities = initialState.velocity();

        this.packedDepths = packedDepths;
        this.packedVelocities = packedVelocities;
        this.auxData = auxData;

        // Calculamos timesteps basándonos en geometría
        this.timestepCount = (geometry.getCellCount() > 0)
                ? packedDepths.length / geometry.getCellCount()
                : 0;
    }

    @Override
    public int getTimestepCount() {
        return timestepCount;
    }

    // --- IMPLEMENTACIÓN IManningResult (Fast Path) ---

    @Override
    public float[] getRawWaterDepthAt(int t) {
        return buildCompositeArray(t, packedDepths, initialDepths);
    }

    @Override
    public float[] getRawVelocityAt(int t) {
        return buildCompositeArray(t, packedVelocities, initialVelocities);
    }

    /**
     * Helper centralizado para construir el array híbrido desde fuentes PLANAS.
     * Combina la ola dinámica (GPU) con el estado base (Initial).
     */
    private float[] buildCompositeArray(int t, float[] packedSource, float[] initialBackground) {
        int cellCount = geometry.getCellCount();
        float[] result = new float[cellCount];

        // 1. Calcular Offset en el array plano
        int offset = t * cellCount;

        // 2. Determinar Frente de Onda (Lógica Smart: 1 celda por paso de tiempo)
        // La GPU nos devuelve datos válidos hasta el frente de onda.
        int copyLength = Math.min(t + 1, cellCount);

        // 3. Copiar ZONA NUEVA (Desde array plano con offset)
        System.arraycopy(packedSource, offset, result, 0, copyLength);

        // 4. Copiar ZONA ANTIGUA (Estado Base Estacionario)
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

        // Rellenado eficiente
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