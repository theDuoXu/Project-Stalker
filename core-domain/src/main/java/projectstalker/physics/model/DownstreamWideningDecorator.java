package projectstalker.physics.model;

import projectstalker.config.RiverConfig;

/**
 * Decorador de Ensanchamiento Longitudinal.
 * <p>
 * Aplica el principio de Acumulación de Caudal (Stream Order).
 * Un río natural tiende a ensancharse a medida que avanza hacia la desembocadura
 * debido a la acumulación de afluentes, escorrentía y aguas subterráneas,
 * independientemente de su pendiente local.
 * <p>
 * Este decorador toma el ancho calculado por otro modelo (ej: basado en pendiente)
 * y lo escala progresivamente en función de la distancia desde la cabecera.
 */
public class DownstreamWideningDecorator implements SpatialModel {

    private final SpatialModel wrappedModel;

    /**
     * @param wrappedModel El modelo base a decorar (ej: SlopeBasedWidthModel).
     */
    public DownstreamWideningDecorator(SpatialModel wrappedModel) {
        this.wrappedModel = wrappedModel;
    }

    @Override
    public float calculate(int cellIndex, RiverConfig config, double driverValue, double noiseFactor) {
        // 1. Obtener el ancho base calculado por la estrategia anterior (Pendiente)
        float baseWidth = wrappedModel.calculate(cellIndex, config, driverValue, noiseFactor);

        // 2. Calcular el progreso longitudinal (0.0 en Nacimiento -> 1.0 en Desembocadura)
        int totalCells = (int) (config.totalLength() / config.spatialResolution());
        // Protección contra división por cero en ríos de 1 celda (tests)
        double progress = (totalCells > 1) ? (double) cellIndex / (totalCells - 1) : 1.0;

        // 3. Determinar el Factor de Crecimiento (Growth Factor) derivado de la longitud
        // Hipótesis: Ríos más largos acumulan más agua y se ensanchan más proporcionalmente.
        // Un río de 10km apenas crece (factor 0.1). Un río de 500km crece mucho (factor 2.0).
        // Fórmula: Growth = Length(km) / 100.0 (Ajustable)
        double lengthInKm = config.totalLength() / 1000.0;

        // Acotamos: Mínimo crece un 5%, máximo crece un 300% (factor 3.0)
        double growthFactor = Math.min(3.0, Math.max(0.05, lengthInKm / 100.0));

        // 4. Calcular el multiplicador local
        // En celda 0: multiplier = 1.0 (Ancho original)
        // En celda final: multiplier = 1.0 + growthFactor
        double multiplier = 1.0 + (growthFactor * progress);

        // 5. Aplicar
        return (float) (baseWidth * multiplier);
    }
}