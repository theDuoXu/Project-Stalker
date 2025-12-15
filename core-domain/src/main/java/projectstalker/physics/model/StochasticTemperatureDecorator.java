package projectstalker.physics.model;

import projectstalker.config.RiverConfig;
import projectstalker.utils.FastNoiseLite;

/**
 * Decorador Estocástico de Temperatura.
 * <p>
 * Añade variabilidad natural (ruido) al perfil térmico para simular efectos locales
 * no modelados explícitamente (sombras parciales, surgencias menores, errores de sensor).
 * Sin esto, los gráficos parecen artificialmente suaves ("demasiado perfectos").
 */
public class StochasticTemperatureDecorator implements TemperatureModel {

    private final TimeEvolutionModel wrappedModel;
    private final float noiseAmplitude;
    private final FastNoiseLite tempNoise;

    public StochasticTemperatureDecorator(TimeEvolutionModel wrappedModel, RiverConfig config) {
        this.wrappedModel = wrappedModel;
        this.noiseAmplitude = config.temperatureNoiseAmplitude();

        // Configuramos un generador de ruido específico para temperatura
        // Usamos seed + 2 para descorrelacionarlo de la geometría del río
        this.tempNoise = new FastNoiseLite((int) config.seed() + 7);
        this.tempNoise.SetNoiseType(FastNoiseLite.NoiseType.Perlin);
        this.tempNoise.SetFrequency(0.1f); // Frecuencia suave
    }

    @Override
    public float[] generateProfile(double currentTimeInSeconds) {
        float[] profile = wrappedModel.generateProfile(currentTimeInSeconds);

        for (int i = 0; i < profile.length; i++) {
            // Obtenemos ruido espacial [-1.0, 1.0]
            float noise = tempNoise.GetNoise(i, 0);

            // Aplicamos amplitud
            profile[i] += noise * noiseAmplitude;
        }

        return profile;
    }
}