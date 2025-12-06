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
 * Pure Flyweight: Utiliza arrays planos COMPACTOS.
 * Los arrays {@code packedDepths} y {@code packedVelocities} no tienen el ancho completo
 * del río, sino solo el {@code activeWidth} (triángulo de cálculo).
 * <p>
 * Esta clase reconstruye el estado completo fusionando la "onda" compacta con el
 * "fondo" estático bajo demanda.
 */
public class FlyweightManningResult implements IManningResult {

    @Getter
    private final RiverGeometry geometry;

    @Getter
    private final long simulationTime;

    // --- ESTADO INTRÍNSECO (Compartido / Base) ---
    private final float[] initialDepths;
    private final float[] initialVelocities;

    // --- ESTADO EXTRÍNSECO (Deltas de GPU - COMPACTOS) ---
    // Layout: Rectángulo de [Timesteps * activeWidth]
    private final float[] packedDepths;
    private final float[] packedVelocities;

    private final int timestepCount;

    /**
     * El ancho de los datos válidos en los arrays packed.
     * Generalmente es min(BatchSize, CellCount).
     */
    private final int activeWidth;

    // Datos auxiliares (Mantenemos estructura original para compatibilidad con módulos químicos CPU)
    private final float[][][] auxData;

    @Builder
    public FlyweightManningResult(RiverGeometry geometry,
                                  long simulationTime,
                                  RiverState initialState,
                                  float[] packedDepths,
                                  float[] packedVelocities,
                                  float[][][] auxData,
                                  int activeWidth) { // <--- Nuevo parámetro crítico
        this.geometry = geometry;
        this.simulationTime = simulationTime;

        // Extracción defensiva inicial (Flyweight Base)
        this.initialDepths = initialState.waterDepth();
        this.initialVelocities = initialState.velocity();

        this.packedDepths = packedDepths;
        this.packedVelocities = packedVelocities;
        this.auxData = auxData;
        this.activeWidth = activeWidth;

        // Calculamos timesteps basándonos en el ancho activo, no en la geometría total
        this.timestepCount = (activeWidth > 0)
                ? packedDepths.length / activeWidth
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
     * Helper centralizado para construir el array híbrido desde fuentes COMPACTAS.
     * Combina la ola dinámica (GPU Compacta) con el estado base (Initial Full).
     */
    private float[] buildCompositeArray(int t, float[] packedSource, float[] initialBackground) {
        int cellCount = geometry.getCellCount();
        float[] result = new float[cellCount];

        // 1. Calcular Offset en el array COMPACTO
        // El salto de fila es 'activeWidth', no 'cellCount'.
        int srcOffset = t * activeWidth;

        // 2. Determinar Frente de Onda (Intersección de Lógica Física y Buffer)
        // La onda llega hasta t+1, pero no puede exceder el ancho del buffer (activeWidth).
        // Tampoco puede exceder el tamaño físico del río (cellCount).
        int waveFront = Math.min(t + 1, activeWidth);

        // Protección extra: waveFront nunca puede ser mayor que cellCount
        // (aunque activeWidth ya debería estar acotado por cellCount, es defensivo).
        waveFront = Math.min(waveFront, cellCount);

        // 3. Copiar ZONA NUEVA (Desde array compacto)
        // Copiamos solo hasta donde llega la onda o el buffer.
        System.arraycopy(packedSource, srcOffset, result, 0, waveFront);

        // 4. Copiar ZONA ANTIGUA (Estado Base Estacionario)
        // Rellenamos el resto del río con el estado inicial.
        int remainingCells = cellCount - waveFront;

        if (remainingCells > 0) {
            // Preservamos el estado inicial en la zona que la ola aún no ha alcanzado.
            // Copiamos desde waveFront hacia waveFront para mantener coherencia espacial.
            System.arraycopy(initialBackground, waveFront, result, waveFront, remainingCells);
        }

        return result;
    }

    // --- IMPLEMENTACIÓN ISimulationResult (Full Object) ---

    @Override
    public RiverState getStateAt(int t) {
        // Reutilizamos la lógica del Fast Path para obtener H y V reconstruidos
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