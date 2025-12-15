package projectstalker.physics.model;

import projectstalker.config.RiverConfig;
import projectstalker.domain.river.RiverGeometry;

/**
 * Decorador de Enfriamiento de Cabecera.
 * <p>
 * Aplica un decaimiento exponencial a la temperatura en el tramo inicial.
 * Utiliza una aproximación de la Ley de Enfriamiento de Newton para evitar
 * transiciones bruscas (codos) en la gráfica.
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
        // 1. Obtener el perfil base
        float[] profile = wrappedModel.generateProfile(currentTimeInSeconds);

        double coolingDist = config.headwaterCoolingDistance(); // Distancia de referencia (L)
        double maxCooling = config.maxHeadwaterCoolingEffect();

        // 2. Definir constante de decaimiento (k)
        // Queremos que en 'coolingDist', el efecto se haya reducido drásticamente (ej: al 2%).
        // e^(-4) ≈ 0.018 (1.8%). Así aseguramos que la "cola" visual coincida con el parámetro.
        final double DECAY_CONSTANT = 4.0;

        // Optimización: Aunque la exponencial nunca llega a 0 absoluto, cortamos
        // cuando el efecto es despreciable para no iterar 100km de río.
        // Calculamos hasta 1.5 veces la distancia configurada para asegurar suavidad total.
        int affectedCells = (int) Math.ceil((coolingDist * 1.5) / spatialResolution);
        int limit = Math.min(profile.length, affectedCells);

        for (int i = 0; i < limit; i++) {
            double position = i * spatialResolution;

            // FÓRMULA: Decaimiento Exponencial (Newton's Law of Heating)
            // Factor = e^(-k * x / L)
            // x=0 -> Factor=1.0 (Máximo enfriamiento)
            // x=L -> Factor=0.018 (Casi temperatura ambiente)
            double decayFactor = Math.exp(-DECAY_CONSTANT * position / coolingDist);

            // Restamos el frío
            profile[i] -= (float) (maxCooling * decayFactor);
        }

        return profile;
    }
}