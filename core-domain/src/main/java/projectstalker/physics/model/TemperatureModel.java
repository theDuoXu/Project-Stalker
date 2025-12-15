package projectstalker.physics.model;

/**
 * Interfaz marcadora para modelos que generan perfiles de TEMPERATURA.
 * <p>
 * Extiende TimeEvolutionModel para heredar la capacidad de generar datos por tiempo,
 * pero añade seguridad semántica: las clases que pidan esto saben que recibirán °C.
 */
@FunctionalInterface
public interface TemperatureModel extends TimeEvolutionModel{
}
