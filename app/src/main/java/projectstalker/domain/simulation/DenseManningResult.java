package projectstalker.domain.simulation;

import lombok.Builder;
import lombok.Value;
import lombok.With;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;

import java.util.List;
import java.util.Optional;

/**
 * Implementación "Densa" (Eager) de los resultados de la simulación.
 * <p>
 * Almacena explícitamente todos los estados calculados en una {@link List} en memoria (Java Heap).
 * Es la implementación utilizada por el solver de CPU y para simulaciones cortas o de validación.
 * <p>
 * <b>Nota sobre Inmutabilidad:</b> Aunque la clase está anotada con {@code @Value},
 * la lista interna {@code states} se trata como mutable para permitir la operación {@code append},
 * lo cual es necesario para la acumulación de resultados en el procesador por lotes de CPU.
 */
@Value
@With
@Builder
public class DenseManningResult implements IManningResult {

    /**
     * La geometría estática sobre la cual se ejecutó la simulación.
     */
    RiverGeometry geometry;

    /**
     * Lista materializada de todos los estados en memoria.
     */
    List<RiverState> states;

    /**
     * Tiempo de cómputo en milisegundos.
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
        return states.size();
    }

    @Override
    public RiverState getStateAt(int index) {
        return states.get(index);
    }

    @Override
    public Optional<RiverState> getFinalState() {
        // Sobreescribimos el default para eficiencia directa sobre la lista
        if (states.isEmpty()) {
            return Optional.empty();
        }
        return Optional.of(states.getLast());
    }

    // --- Métodos Específicos de la Implementación Densa (Mutación/Append) ---

    /**
     * Añade nuevos estados al final de la lista existente.
     * Útil para la ejecución por lotes en CPU donde se acumulan resultados.
     */
    public void appendNewStates(List<RiverState> newStates) {
        if (newStates == null || newStates.isEmpty()) return;

        // Validación de consistencia geométrica
        if (newStates.getFirst().waterDepth().length != getGeometry().getCellCount()) {
            throw new IllegalArgumentException("Error de consistencia: El nuevo estado tiene dimensiones diferentes a la geometría del río.");
        }

        // Mutación de la lista subyacente
        states.addAll(newStates);
    }

    /**
     * Fusiona otro resultado denso en este.
     * Valida que la geometría sea idéntica antes de fusionar.
     */
    public DenseManningResult appendNewManningSimulationResult(DenseManningResult newResult) {
        if (!newResult.getGeometry().equals(this.geometry)) {
            throw new IllegalArgumentException("No se pueden fusionar resultados: La geometría del río difiere.");
        }
        this.appendNewStates(newResult.getStates());
        return this;
    }

    @Override
    public float[] getRawWaterDepthAt(int logicalT) {
        return states.get(logicalT).waterDepth();
    }

    @Override
    public float[] getRawVelocityAt(int logicalT) {
        return states.get(logicalT).velocity();
    }
}