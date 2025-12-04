package projectstalker.domain.simulation;

import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;

import java.util.Optional;

/**
 * Contrato común para los resultados de la simulación hidrológica.
 * <p>
 * Permite desacoplar la forma en que se almacenan los datos (Dense vs Flyweight)
 * de cómo se consumen (UI, Exportadores, Tests).
 */
public interface ISimulationResult {

    /**
     * Obtiene la geometría del río asociada a esta simulación.
     */
    RiverGeometry getGeometry();

    /**
     * Devuelve el tiempo total que tardó el cálculo (métrica de rendimiento).
     */
    long getSimulationTime();

    /**
     * Devuelve el número total de pasos de tiempo (timesteps) disponibles.
     */
    int getTimestepCount();

    /**
     * Obtiene el estado del río en un momento específico.
     * <p>
     * En implementaciones <b>Dense</b>, esto devuelve un objeto ya existente en memoria.
     * En implementaciones <b>Flyweight</b>, esto puede generar el objeto al vuelo combinando
     * datos comprimidos y estado base.
     *
     * @param index Índice del paso de tiempo (0 a getTimestepCount() - 1).
     * @return El estado del río en ese instante.
     */
    RiverState getStateAt(int index);

    /**
     * Helper para obtener el último estado (útil para encadenar simulaciones).
     */
    default Optional<RiverState> getFinalState() {
        if (getTimestepCount() == 0) {
            return Optional.empty();
        }
        return Optional.of(getStateAt(getTimestepCount() - 1));
    }
}