package projectstalker.domain.simulation;

import lombok.Builder;
import lombok.Value;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;

import java.util.List;
import java.util.Optional;

/**
 * Implementación "Chunked" (Paginada) de los resultados.
 * <p>
 * ALMACENAMIENTO:
 * Diseñada para simulaciones masivas (Big Data / Long Running) que exceden el tamaño
 * máximo de un array de Java (~2GB) o que se reciben por lotes desde la GPU.
 * <p>
 * ESTRUCTURA:
 * - Lista de "Chunks" (Páginas): {@code List<float[]>}.
 * - Soporta "Striding" (Saltos temporales) de forma nativa.
 * <p>
 * ARITMÉTICA DE ACCESO:
 * Logical Time -> [Stride] -> Stored Index -> [Paginación] -> {Chunk Index + Offset}
 */
@Value
@Builder
public class ChunkedManningResult implements IManningResult {

    /**
     * Geometría del río.
     */
    RiverGeometry geometry;

    /**
     * Lista de páginas (chunks) de Profundidad.
     * Cada elemento es un float[] que contiene 'stepsPerChunk' pasos de tiempo.
     */
    List<float[]> depthChunks;

    /**
     * Lista de páginas (chunks) de Velocidad.
     */
    List<float[]> velocityChunks;

    /**
     * Capacidad de almacenamiento de cada chunk (en pasos de tiempo GUARDADOS).
     * <p>
     * CRÍTICO: Todos los chunks (excepto posiblemente el último) deben tener este tamaño
     * para que la matemática de acceso directo funcione.
     */
    int stepsPerChunk;

    /**
     * Factor de salto.
     * 1 = Se guardó todo.
     * 10 = Se guardó 1 de cada 10 pasos lógicos.
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

    @Override
    public RiverState getStateAt(int logicalT) {
        // 1. Obtener los arrays crudos (Fast Path interno)
        float[] h = getRawWaterDepthAt(logicalT);
        float[] v = getRawVelocityAt(logicalT);

        // 2. Rellenar variables dummy
        int cells = geometry.getCellCount();
        float[] empty = new float[cells];

        // 3. Construir DTO
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

    // --- MÉTODOS "FAST PATH" (Acceso directo a memoria) ---
    @Override
    public float[] getRawWaterDepthAt(int logicalT) {
        // 1. Calcular dirección de memoria virtual
        MemoryAddress addr = calculateAddress(logicalT);

        // 2. Acceder al Chunk correcto
        float[] chunk = depthChunks.get(addr.chunkIndex);

        // 3. Copiar la rebanada (Slice)
        int cellCount = geometry.getCellCount();
        float[] slice = new float[cellCount];

        // Offset = paso_local * celdas
        int memoryOffset = addr.localStepIndex * cellCount;

        System.arraycopy(chunk, memoryOffset, slice, 0, cellCount);
        return slice;
    }
    @Override
    public float[] getRawVelocityAt(int logicalT) {
        MemoryAddress addr = calculateAddress(logicalT);
        float[] chunk = velocityChunks.get(addr.chunkIndex);

        int cellCount = geometry.getCellCount();
        float[] slice = new float[cellCount];
        int memoryOffset = addr.localStepIndex * cellCount;

        System.arraycopy(chunk, memoryOffset, slice, 0, cellCount);
        return slice;
    }

    // --- Helpers de Direccionamiento ---

    /**
     * "Puntero" interno para localizar el dato.
     */
    private record MemoryAddress(int chunkIndex, int localStepIndex) {}

    private MemoryAddress calculateAddress(int logicalT) {
        // 1. Clamping (Asegurar límites lógicos)
        if (logicalT < 0) logicalT = 0;
        if (logicalT >= logicTimestepCount) logicalT = logicTimestepCount - 1;

        // 2. Mapping Lógico -> Físico (Global Storage Index)
        int globalStorageIndex = logicalT / strideFactor;

        // 3. Paginación (Chunking)
        int chunkIndex = globalStorageIndex / stepsPerChunk;
        int localStepIndex = globalStorageIndex % stepsPerChunk;

        // 4. Protección contra desbordamiento del último chunk
        // Si el último chunk está incompleto y el cálculo matemático apunta fuera,
        // retrocedemos al último dato válido disponible.
        if (chunkIndex >= depthChunks.size()) {
            chunkIndex = depthChunks.size() - 1;
            // En el último chunk, tomamos el último paso disponible
            // (Esta lógica asume que el user no pide más allá de logicTimestepCount,
            // pero es una salvaguarda de seguridad).
            int maxStepsInLastChunk = (depthChunks.get(chunkIndex).length / geometry.getCellCount());
            localStepIndex = maxStepsInLastChunk - 1;
        }

        return new MemoryAddress(chunkIndex, localStepIndex);
    }
}