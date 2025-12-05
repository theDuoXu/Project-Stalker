package projectstalker.domain.simulation;

/**
 * Interfaz especializada para resultados de simulaciones hidráulicas (Manning).
 * <p>
 * Expone métodos de "Fast Path" para acceder directamente a los arrays primitivos
 * de profundidad y velocidad sin el overhead de instanciar objetos RiverState.
 * Ideal para renderizado gráfico (OpenGL/JavaFX) y exportación de datos.
 */
public interface IManningResult extends ISimulationResult {

    /**
     * Obtiene el array crudo de profundidades [m] para el instante lógico 't'.
     * @return float[] array copiado o referenciado con los datos.
     */
    float[] getRawWaterDepthAt(int logicalT);

    /**
     * Obtiene el array crudo de velocidades [m/s] para el instante lógico 't'.
     * @return float[] array copiado o referenciado con los datos.
     */
    float[] getRawVelocityAt(int logicalT);
}