package projectstalker.physics.model;

import projectstalker.config.RiverConfig;
import projectstalker.domain.river.RiverGeometry;

/**
 * Decorador de Enfriamiento de Cabecera.
 * <p>
 * Modifica el perfil de temperatura simulando que el agua nace más fría.
 * <p>
 * Mejoras aplicadas:
 * 1. <b>Curva Cosenoidal (SmoothStep):</b> Elimina la agresividad inicial. La transición es suave al principio y al final.
 * 2. <b>Protección Anti-Congelación:</b> Se asegura de que la resta nunca baje de 0.1°C.
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
        // 1. Obtener el perfil base (Temperatura ambiente/estacional)
        float[] profile = wrappedModel.generateProfile(currentTimeInSeconds);

        double coolingDist = config.headwaterCoolingDistance();
        double maxCooling = config.maxHeadwaterCoolingEffect();

        // Si la distancia es 0 o negativa, no hacemos nada
        if (coolingDist <= 0) return profile;

        // Iteramos solo hasta la distancia de efecto
        int affectedCells = (int) Math.ceil(coolingDist / spatialResolution);
        int limit = Math.min(profile.length, affectedCells);

        for (int i = 0; i < limit; i++) {
            double position = i * spatialResolution;

            // Progreso normalizado de 0.0 (nacimiento) a 1.0 (fin del efecto)
            double progress = position / coolingDist;

            // FÓRMULA: Interpolación Cosenoidal (Half-Cosine / SmoothStep)
            // En x=0 -> cos(0)=1 -> (1+1)/2 = 1.0 (100% efecto)
            // En x=1 -> cos(pi)=-1 -> (1-1)/2 = 0.0 (0% efecto)
            // Esto genera una curva mucho más suave que la lineal o la exponencial agresiva.
            double smoothFactor = 0.5 * (1.0 + Math.cos(Math.PI * progress));

            // Calculamos la temperatura propuesta
            float currentTemp = profile[i];
            float coolingAmount = (float) (maxCooling * smoothFactor);
            float newTemp = currentTemp - coolingAmount;

            // CORRECCIÓN FÍSICA: El agua líquida no baja de 0°C (aprox)
            // Si hacía 2°C y restamos 4°C, nos quedamos en 0.1°C, no en -2°C.
            profile[i] = Math.max(0.1f, newTemp);
        }

        return profile;
    }
}