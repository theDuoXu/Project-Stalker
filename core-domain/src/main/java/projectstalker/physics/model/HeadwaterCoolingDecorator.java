package projectstalker.physics.model;

import projectstalker.config.RiverConfig;
import projectstalker.domain.river.RiverGeometry;

/**
 * Decorador de Enfriamiento de Cabecera.
 * <p>
 * Modifica el perfil de temperatura aplicando un gradiente negativo en el nacimiento del río.
 * Simula el hecho de que el agua de origen (manantial, deshielo) suele estar más fría
 * que la temperatura de equilibrio ambiental, calentándose a medida que fluye río abajo.
 */
public class HeadwaterCoolingDecorator implements TemperatureModel {

    private final TemperatureModel wrappedModel;
    private final RiverConfig config;
    private final double spatialResolution;

    public HeadwaterCoolingDecorator(TemperatureModel wrappedModel, RiverConfig config, RiverGeometry geometry) {
        this.wrappedModel = wrappedModel;
        this.config = config;
        this.spatialResolution = geometry.getSpatialResolution();
    }

    @Override
    public float[] generateProfile(double currentTimeInSeconds) {
        // 1. Obtener el perfil base (del modelo envuelto)
        float[] profile = wrappedModel.generateProfile(currentTimeInSeconds);

        // 2. Aplicar el efecto de enfriamiento solo donde sea relevante
        double coolingDist = config.headwaterCoolingDistance();
        double maxCooling = config.maxHeadwaterCoolingEffect();

        // Optimización: Solo iteramos hasta la distancia de enfriamiento, no todo el río
        int affectedCells = (int) Math.ceil(coolingDist / spatialResolution);
        int limit = Math.min(profile.length, affectedCells);

        for (int i = 0; i < limit; i++) {
            double position = i * spatialResolution;

            // Factor de 1.0 (en el nacimiento) a 0.0 (al final de la distancia de enfriamiento)
            double gradientFactor = 1.0 - (position / coolingDist);

            if (gradientFactor > 0) {
                // Restamos temperatura (Enfriamiento)
                profile[i] -= (float) (maxCooling * gradientFactor);
            }
        }

        return profile;
    }
}