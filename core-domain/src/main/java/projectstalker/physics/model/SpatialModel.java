package projectstalker.physics.model;

import projectstalker.config.RiverConfig;

/**
 * Define el contrato para modelos físicos que calculan propiedades del río
 * basándose en su ubicación espacial y condiciones locales.
 * <p>
 * A diferencia de los modelos temporales, estos modelos son deterministas
 * respecto a la geometría y configuración estática.
 */
@FunctionalInterface
public interface SpatialModel {
    double EPSILON = 0.0001;

    /**
     * Calcula el valor de una propiedad física en un punto específico.
     *
     * @param cellIndex   El índice de la celda actual (para efectos longitudinales).
     * @param config      La configuración global del río (valores base).
     * @param driverValue El valor físico conductor principal (ej: Pendiente Local).
     * @param noiseFactor Un valor de ruido normalizado (típicamente de -1.0 a 1.0) para variabilidad local.
     * @return El valor calculado de la propiedad (ej: Ancho en metros, Manning, etc.).
     */
    float calculate(int cellIndex, RiverConfig config, double driverValue, double noiseFactor);

}