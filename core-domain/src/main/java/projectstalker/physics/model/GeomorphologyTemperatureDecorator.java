package projectstalker.physics.model;

import projectstalker.config.RiverConfig;
import projectstalker.domain.river.RiverGeometry;

/**
 * Decorador Geomoforfológico de Temperatura.
 * <p>
 * Ajusta la temperatura basándose en la forma física del cauce:
 * <ul>
 * <li><b>Efecto Ancho (Insolación):</b> Tramos anchos aumentan la superficie de exposición solar,
 * incrementando la temperatura relativa.</li>
 * <li><b>Efecto Pendiente (Evaporación/Mezcla):</b> Tramos de alta pendiente aumentan
 * la turbulencia y la evaporación (proceso endotérmico), tendiendo a refrigerar levemente el agua.</li>
 * </ul>
 */
public class GeomorphologyTemperatureDecorator implements TemperatureModel {

    private final TemperatureModel wrappedModel;
    private final RiverConfig config;
    private final RiverGeometry geometry;

    public GeomorphologyTemperatureDecorator(TemperatureModel wrappedModel, RiverConfig config, RiverGeometry geometry) {
        this.wrappedModel = wrappedModel;
        this.config = config;
        this.geometry = geometry;
    }

    @Override
    public float[] generateProfile(double currentTimeInSeconds) {
        // 1. Obtener perfil base (Clima + Cabecera)
        float[] profile = wrappedModel.generateProfile(currentTimeInSeconds);

        // 2. Aplicar correcciones físicas
        int cells = geometry.getCellCount();

        // Pre-cálculo de constantes para evitar accesos repetidos
        double baseWidth = config.baseWidth();
        double avgSlope = config.averageSlope();
        double heatingFactor = config.widthHeatingFactor();
        double coolingFactor = config.slopeCoolingFactor();

        for (int i = 0; i < cells; i++) {
            // A. Efecto de Ancho (Heating)
            // Si el río es más ancho que la base, se calienta extra.
            double currentWidth = geometry.getWidthAt(i);
            double widthRatio = currentWidth / baseWidth;

            // Solo aplicamos calentamiento si es más ancho de lo normal (ratio > 1.0)
            double widthEffect = (widthRatio > 1.0)
                    ? (widthRatio - 1.0) * heatingFactor
                    : 0.0;

            // B. Efecto de Pendiente (Cooling)
            // Si la pendiente es mayor a la media, hay más turbulencia/evaporación -> Enfría.
            double currentSlope = geometry.getBedSlopeAt(i);
            double slopeRatio = currentSlope / avgSlope;

            double slopeEffect = (slopeRatio > 1.0)
                    ? (slopeRatio - 1.0) * coolingFactor
                    : 0.0;

            // Aplicar cambios
            profile[i] += (float) (widthEffect - slopeEffect);
        }

        return profile;
    }
}