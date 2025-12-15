package projectstalker.physics.model;

import projectstalker.config.RiverConfig;
import projectstalker.domain.river.RiverGeometry;

/**
 * Modelo de Evolución Temporal del pH.
 * <p>
 * Simula el ciclo biogeoquímico diario:
 * <ul>
 * <li>El modelo Espacial (Factory) define la línea base geológica.</li>
 * <li>Este modelo aplica la oscilación sinusoidal debida a la fotosíntesis.</li>
 * </ul>
 */
public class RiverPhModel implements TimeEvolutionModel {

    private static final double SECONDS_IN_A_DAY = 86400.0;

    // Desfase para que el pico de pH sea por la tarde (ej: 15:00 - 16:00)
    private final double PHASE_SHIFT_HOURS;

    private final RiverConfig config;
    private final RiverGeometry geometry;
    private final TemperatureModel temperatureModel;

    public RiverPhModel(RiverConfig config, RiverGeometry geometry, TemperatureModel temperatureModel) {
        this.config = config;
        this.geometry = geometry;
        this.PHASE_SHIFT_HOURS = config.riverPhaseShiftHours();
        this.temperatureModel = temperatureModel;
    }

    /**
     * Calcula el perfil de pH influenciado por la geología (base), el ciclo solar (tiempo)
     * y la actividad biológica (modulada por temperatura y tasa de descomposición).
     *
     * @param currentTimeInSeconds Tiempo actual.
     * @return Perfil de pH dinámico.
     */
    @Override
    public float[] generateProfile(double currentTimeInSeconds) {
        // Conseguir el estado de la temperatura actual
        float[] temperatureProfile = this.temperatureModel.generateProfile(currentTimeInSeconds);
        return generateProfile(currentTimeInSeconds, temperatureProfile);
    }

    /**
     * Versión con temperaturas precalculadas
     */
    public float[] generateProfile(double currentTimeInSeconds, float[] temperatureProfile) {
        final int cellCount = geometry.getCellCount();
        final float[] currentPhProfile = new float[cellCount];

        // 1. Datos Estáticos (Geometría y Configuración)
        float[] baselinePh = this.geometry.getPhProfile(); // El pH geológico base (SpatialModel)
        float[] decayRates = this.geometry.getBaseDecayCoefficientAt20C(); // Proxy de actividad biológica
        float baseDecay = this.config.baseDecayRateAt20C();

        // 2. Factor Ciclo Solar (Global) [-1.0 a 1.0]
        double dailyCycle = calculateDailyCycle(currentTimeInSeconds);

        // Constante térmica biológica (Theta): Típicamente 1.047 a 1.072 para tasas biológicas.
        // Usamos 1.072 para que el efecto de la temperatura sea notable.
        final double THETA = 1.072;

        // 3. Cálculo Celda a Celda
        for (int i = 0; i < cellCount; i++) {
            // A. Corrección por Temperatura (Arrhenius simplificado)
            // Si T > 20°C, el metabolismo se acelera -> Mayor amplitud de pH.
            // Si T < 20°C, el metabolismo se frena.
            double currentTemp = temperatureProfile[i];
            double tempCorrection = Math.pow(THETA, currentTemp - 20.0);

            // B. Factor de Biomasa Local
            // Si esta celda tiene un DecayRate alto (aguas estancadas/sucias),
            // asumimos más algas/bacterias -> Mayor oscilación de pH.
            // Normalizamos respecto al valor base configurado para evitar explosiones.
            double bioActivityFactor = (baseDecay > 0) ? (decayRates[i] / baseDecay) : 1.0;

            // Acotamos el factor biológico para seguridad (0.5x a 3.0x)
            bioActivityFactor = Math.max(0.5, Math.min(3.0, bioActivityFactor));

            // C. Calcular Amplitud Local Dinámica
            // Amplitud Base * Factor Temperatura * Factor Biológico
            double localAmplitude = (config.phVariability() * 0.5) * tempCorrection * bioActivityFactor;

            // D. Aplicar Oscilación
            // pH(t) = BaseGeológica + AmplitudDinámica * Seno(t)
            currentPhProfile[i] = (float) (baselinePh[i] + (localAmplitude * dailyCycle));
        }

        return currentPhProfile;
    }

    private double calculateDailyCycle(double time) {
        // Normalizamos el tiempo al día actual
        double timeOfDay = time % SECONDS_IN_A_DAY;

        // Ajustamos para que el pico coincida con PHASE_SHIFT_HOURS
        // La función Sin tiene pico en PI/2.
        double omega = (2.0 * Math.PI) / SECONDS_IN_A_DAY;
        double shiftInSeconds = (PHASE_SHIFT_HOURS - 6.0) * 3600.0; // Ajuste empírico

        return Math.sin(omega * (timeOfDay - shiftInSeconds));
    }
}