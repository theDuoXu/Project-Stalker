package projectstalker.physics.model;

import lombok.extern.slf4j.Slf4j;
import projectstalker.config.RiverConfig;
import projectstalker.domain.river.RiverGeometry;

import java.util.Arrays;

/**
 * Modelo Base de Temperatura.
 * <p>
 * Responsabilidad: Calcular la temperatura "ambiental" del agua basándose exclusivamente
 * en los ciclos temporales (Estacional y Diario).
 * <p>
 * Este modelo ignora la geografía; asume que todo el cuerpo de agua está expuesto
 * a las mismas condiciones climáticas globales. Sirve como lienzo base sobre el cual
 * los decoradores aplicarán los efectos locales.
 */
@Slf4j
public class ClimatologicalTemperatureModel implements TemperatureModel {

    private static final double SECONDS_IN_A_DAY = 24.0 * 3600.0;
    private static final double DAYS_IN_A_YEAR = 365.25;

    private final RiverConfig config;
    private final int cellCount;

    public ClimatologicalTemperatureModel(RiverConfig config, RiverGeometry geometry) {
        this.config = config;
        this.cellCount = geometry.getCellCount();
    }

    @Override
    public float[] generateProfile(double currentTimeInSeconds) {
        // 1. Calcular la temperatura global para este instante
        float globalBaseTemp = calculateGlobalBaseTemperature(currentTimeInSeconds);

        // 2. Crear el perfil plano (inicialmente, todo el río tiene la misma T)
        float[] temperatureProfile = new float[cellCount];
        Arrays.fill(temperatureProfile, globalBaseTemp);

        return temperatureProfile;
    }

    private float calculateGlobalBaseTemperature(double currentTimeInSeconds) {
        // Ciclo Anual (Estacional)
        final double dayOfYear = (currentTimeInSeconds / SECONDS_IN_A_DAY) % DAYS_IN_A_YEAR;
        final double seasonalCycle = Math.sin((dayOfYear / DAYS_IN_A_YEAR) * 2.0 * Math.PI);
        final double baseSeasonalTemp = config.averageAnnualTemperature() + (config.seasonalTempVariation() * seasonalCycle);

        // Ciclo Diario (Día/Noche)
        final double secondOfDay = currentTimeInSeconds % SECONDS_IN_A_DAY;
        // Ajuste: El pico de calor suele ser post-mediodía (inercia térmica), pero mantenemos la sinusoide simple por ahora
        final double dailyCycle = Math.sin((secondOfDay / SECONDS_IN_A_DAY) * 2.0 * Math.PI);

        return (float) (baseSeasonalTemp + (config.dailyTempVariation() * dailyCycle));
    }
}