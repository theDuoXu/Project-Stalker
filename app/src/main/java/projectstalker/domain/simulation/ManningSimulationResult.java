package projectstalker.domain.simulation; // Un buen paquete para los resultados

import lombok.Builder;
import lombok.Value;
import projectstalker.config.SimulationConfig;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;

import java.util.Arrays;
import java.util.List;
import java.util.Optional;

/**
 * Encapsula el resultado completo de una ejecución de la simulación hidrológica.
 * <p>
 * Este objeto es inmutable y contiene tanto la geometría estática del río
 * utilizada en la simulación como la secuencia de estados del río a lo largo del tiempo.
 */
@Value
@Builder
public class ManningSimulationResult {

    /**
     * La geometría del río sobre la cual se ejecutó la simulación.
     */
    RiverGeometry geometry;

    /**
     * Una lista cronológica de los estados del río, donde cada estado representa un paso de tiempo.
     */
    List<RiverState> states;


    /**
     * Devuelve el número total de pasos de tiempo (timesteps) registrados en el resultado.
     *
     * @return el tamaño de la lista de estados.
     */
    public int getTimestepCount() {
        return states.size();
    }

    /**
     * Obtiene el estado del río en un índice de tiempo específico.
     *
     * @param index el índice del paso de tiempo.
     * @return el RiverState en esa posición.
     * @throws IndexOutOfBoundsException si el índice está fuera de rango.
     */
    public RiverState getStateAt(int index) {
        return states.get(index);
    }

    /**
     * Devuelve el estado final de la simulación.
     *
     * @return un Optional que contiene el último RiverState, o un Optional vacío si no hay estados.
     */
    public Optional<RiverState> getFinalState() {
        if (states.isEmpty()) {
            return Optional.empty();
        }
        return Optional.of(states.getLast());
    }

    public void appendNewStates(List<RiverState> newStates) {
        if (newStates.getFirst().waterDepth().length != states.getFirst().waterDepth().length) {
            throw new IllegalArgumentException("Es posible que el nuevo estado no se haya generado con la misma geometría del río");
        }
        states.addAll(newStates);
    }
}