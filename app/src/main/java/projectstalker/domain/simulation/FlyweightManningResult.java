package projectstalker.domain.simulation;

import lombok.Builder;
import lombok.Getter;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;

import java.util.Arrays;

/**
 * Implementación "Flyweight" (Lazy) de los resultados de la simulación.
 * <p>
 * Combina resultados dinámicos de la GPU (Extrinsic State) con el estado base del río (Intrinsic State).
 * Soporta submuestreo (Stride) para ahorrar memoria en simulaciones largas.
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
    // Puede estar compactado si stride > 1
    private final float[][][] gpuPackedData;

    // Datos auxiliares (Temp, pH) - Normalmente resolución completa
    private final float[][][] auxData;

    // Factor de submuestreo (1 = Todos los pasos, N = 1 de cada N)
    private final int stride;

    /**
     * Constructor personalizado con soporte para Stride.
     */
    @Builder
    public FlyweightManningResult(RiverGeometry geometry,
                                  long simulationTime,
                                  RiverState initialState,
                                  float[][][] gpuPackedData,
                                  float[][][] auxData,
                                  int stride) {
        this.geometry = geometry;
        this.simulationTime = simulationTime;

        // Extracción defensiva inicial (Flyweight Base)
        this.initialDepths = initialState.waterDepth();
        this.initialVelocities = initialState.velocity();

        this.gpuPackedData = gpuPackedData;
        this.auxData = auxData;

        // Validación para evitar división por cero
        this.stride = Math.max(1, stride);
    }

    // Sobrecarga para compatibilidad (Stride = 1)
    public FlyweightManningResult(RiverGeometry geometry, long time, RiverState initial, float[][][] gpu, float[][][] aux) {
        this(geometry, time, initial, gpu, aux, 1);
    }

    @Override
    public int getTimestepCount() {
        // Devolvemos el número de pasos lógicos representados
        return gpuPackedData.length * stride;
    }

    /**
     * Reconstruye el estado del río en el tiempo 't' bajo demanda.
     * Si hay stride, devuelve el snapshot más cercano (Sample & Hold).
     */
    @Override
    public RiverState getStateAt(int t) {
        int cellCount = geometry.getCellCount();

        float[] h = new float[cellCount];
        float[] v = new float[cellCount];

        // 1. MAPEO DE TIEMPO A ALMACENAMIENTO (Lógica Stride)
        // t es el paso de tiempo lógico (0...TotalSteps)
        // storageIndex es el índice físico en el array GPU (0...PackedLength)
        int storageIndex = t / stride;

        // Protección contra desbordamiento (Clamping al último frame disponible)
        if (storageIndex >= gpuPackedData.length) {
            storageIndex = gpuPackedData.length - 1;
        }

        // 2. ZONA NUEVA (GPU Data - Onda Dinámica Activa)
        float[] gpuH = gpuPackedData[storageIndex][0];
        float[] gpuV = gpuPackedData[storageIndex][1];

        // Determinamos cuánto datos válidos nos dio la GPU.
        // En modo Smart: gpuH.length crece con el tiempo (Triangular).
        // En modo Full: gpuH.length == cellCount (Rectangular).
        // Usamos gpuH.length directamente, es la fuente de verdad.
        int copyLength = Math.min(gpuH.length, cellCount);

        System.arraycopy(gpuH, 0, h, 0, copyLength);
        System.arraycopy(gpuV, 0, v, 0, copyLength);

        // 3. ZONA ANTIGUA (Estado Base Estacionario)
        int remainingCells = cellCount - copyLength;

        if (remainingCells > 0) {
            // Lógica Correcta: Copia Estática Local.
            // Copiamos la profundidad que tenía esta celda en el estado base.
            // NO desplazamos desde el inicio (srcPos != 0), mantenemos srcPos == destPos.
            System.arraycopy(initialDepths, copyLength, h, copyLength, remainingCells);
            System.arraycopy(initialVelocities, copyLength, v, copyLength, remainingCells);
        }

        // 4. CONSTRUCCIÓN DEL ESTADO AUXILIAR
        // Asumimos que auxData (CPU) tiene resolución completa (índice t).
        // Si no (si también viniera compactado), habría que usar storageIndex.
        // Aplicamos protección de límites por seguridad.
        int auxIndex = Math.min(t, (auxData != null ? auxData.length - 1 : 0));

        float t_val = (auxData != null && auxIndex >= 0) ? auxData[auxIndex][0][0] : 0f;
        float ph_val = (auxData != null && auxIndex >= 0) ? auxData[auxIndex][1][0] : 0f;

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