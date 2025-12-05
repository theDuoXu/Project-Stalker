package projectstalker.domain.simulation;

import lombok.Builder;
import lombok.Value;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;

import java.util.Arrays;
import java.util.Optional;

/**
 * Implementación "Strided" (Con Saltos) optimizada.
 * <p>
 * ALMACENAMIENTO:
 * Guarda los resultados en dos grandes arrays primitivos planos (packed):
 * - {@code packedDepths}
 * - {@code packedVelocities}
 * <p>
 * Esto evita el overhead de crear objetos {@link RiverState} intermedios y
 * arrays vacíos para las variables químicas no simuladas.
 */
@Value
@Builder
public class StridedManningResult implements IManningResult {

    /**
     * Geometría del río.
     */
    RiverGeometry geometry;

    /**
     * Array PLANO que contiene todas las profundidades concatenadas.
     * <p>
     * Layout: [Step0_Cell0, Step0_Cell1... | Step1_Cell0... ]
     * Tamaño total = (storedSteps * cellCount).
     */
    float[] packedDepths;

    /**
     * Array PLANO que contiene todas las velocidades concatenadas.
     */
    float[] packedVelocities;

    /**
     * El factor de salto. Ej: 10 significa que un avance de 1 índice en los arrays packed
     * equivale a 10 pasos de tiempo lógico.
     */
    int strideFactor;

    /**
     * Cantidad total de pasos de tiempo de la simulación lógica original.
     */
    int logicTimestepCount;

    /**
     * Tiempo de cómputo.
     */
    long simulationTime;

    // --- Implementación de ISimulationResult ---

    @Override
    public RiverGeometry getGeometry() {
        return geometry;
    }

    @Override
    public long getSimulationTime() {
        return simulationTime;
    }

    @Override
    public int getTimestepCount() {
        return logicTimestepCount;
    }

    /**
     * Recupera el estado completo envolviéndolo en un RiverState.
     * <p>
     * ADVERTENCIA DE RENDIMIENTO: Este método instancia un nuevo RiverState,
     * lo cual provoca la clonación defensiva de los arrays y la creación de
     * arrays vacíos para temperatura/pH. Usar con moderación.
     */
    @Override
    public RiverState getStateAt(int logicalT) {
        // 1. Obtener los arrays crudos de H y V
        float[] h = getRawWaterDepthAt(logicalT);
        float[] v = getRawVelocityAt(logicalT);

        // 2. Generar arrays "dummy" para las variables no simuladas
        // (RiverState exige que tengan el mismo tamaño)
        int cells = geometry.getCellCount();
        float[] empty = new float[cells]; // Todo a 0.0f por defecto

        // 3. Construir el DTO (Aquí ocurre la validación y clonación defensiva de RiverState)
        // Nota: Pasamos 'empty' clonado implícitamente por RiverState, es el costo de la inmutabilidad.
        return RiverState.builder()
                .waterDepth(h)
                .velocity(v)
                .temperature(empty)
                .ph(empty)
                .contaminantConcentration(empty)
                .build();
    }

    @Override
    public Optional<RiverState> getFinalState() {
        if (logicTimestepCount == 0) return Optional.empty();
        return Optional.of(getStateAt(logicTimestepCount - 1));
    }

    // --- MÉTODOS "FAST PATH" (Sin instanciar RiverState) ---

    /**
     * Obtiene directamente el array de PROFUNDIDADES para el tiempo lógico dado.
     * <p>
     * Realiza una copia del segmento de memoria correspondiente (Slice).
     * Más eficiente que getStateAt() si solo necesitas graficar H.
     */
    public float[] getRawWaterDepthAt(int logicalT) {
        int storageIndex = calculateStorageIndex(logicalT);
        int cellCount = geometry.getCellCount();
        int offset = storageIndex * cellCount;

        float[] slice = new float[cellCount];
        System.arraycopy(packedDepths, offset, slice, 0, cellCount);
        return slice;
    }

    /**
     * Obtiene directamente el array de VELOCIDADES para el tiempo lógico dado.
     */
    public float[] getRawVelocityAt(int logicalT) {
        int storageIndex = calculateStorageIndex(logicalT);
        int cellCount = geometry.getCellCount();
        int offset = storageIndex * cellCount;

        float[] slice = new float[cellCount];
        System.arraycopy(packedVelocities, offset, slice, 0, cellCount);
        return slice;
    }

    // --- Helpers Internos ---

    private int calculateStorageIndex(int logicalT) {
        // Clamping (Saturación)
        if (logicalT < 0) logicalT = 0;
        if (logicalT >= logicTimestepCount) logicalT = logicTimestepCount - 1;

        // Conversión Lógico -> Físico (Striding)
        int storageIndex = logicalT / strideFactor;

        // Protección extra por si el cálculo de stride no fue exacto al guardar
        int maxStoredSteps = packedDepths.length / geometry.getCellCount();
        if (storageIndex >= maxStoredSteps) {
            storageIndex = maxStoredSteps - 1;
        }
        return storageIndex;
    }
}