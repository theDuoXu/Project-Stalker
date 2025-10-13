package projectstalker.config;


/**
 * Un objeto de valor inmutable para contener todos los parámetros de configuración
 * necesarios para la generación procedural de un río.
 * <p>
 * Esta clase agrupa todas las variables de alto nivel que definen la morfología,
 * hidrología y las propiedades físico-químicas básicas del río a simular.
 *
 * @param seed Semilla para la generación de ruido, para resultados reproducibles.
 * @param totalLength Longitud total del río en metros.
 * @param spatialResolution Resolución espacial (dx) en metros.
 * @param initialElevation Altitud inicial del río en metros.
 * @param averageSlope Pendiente media del río (adimensional, ej: 0.001 para 1m de caída cada km).
 * @param slopeVariability Factor de variabilidad para la pendiente (adimensional).
 * @param baseWidth Ancho base del fondo del río en metros.
 * @param widthVariability Variación máxima del ancho en metros (ej: 5.0 para +/- 5m).
 * @param baseSideSlope Pendiente base de los taludes (adimensional).
 * @param sideSlopeVariability Variación máxima de la pendiente de los taludes.
 * @param baseManning Coeficiente de Manning base.
 * @param manningVariability Variación máxima del coeficiente de Manning.
 *
 * @param baseDecayRateAt20C Coeficiente de reacción/descomposición base (k) a 20°C, en unidades de s⁻¹.
 * @param decayRateVariability Variación máxima del coeficiente de reacción.
 *
 * @param baseTemperature Temperatura media diaria del agua en grados Celsius (°C).
 * @param dailyTempVariation Amplitud de la variación diaria de temperatura en °C (ej: 3.0 para +/- 3°C).
 * @param basePh pH base del agua (ej: 7.5).
 * @param phVariability Variación máxima del pH a lo largo del río.
 * @param seasonalTempVariation Amplitud máxima, temperatura anual
 */
public record RiverConfig(
        long seed,
        double totalLength,
        double spatialResolution,
        double initialElevation,
        double averageSlope,
        double slopeVariability,
        double baseWidth,
        double widthVariability,
        double baseSideSlope,
        double sideSlopeVariability,
        double baseManning,
        double manningVariability,

        // --- Parámetros de Reacción ---
        double baseDecayRateAt20C,
        double decayRateVariability,

        // --- Parámetros de Calidad de Agua ---
        double baseTemperature,
        double dailyTempVariation,
        double seasonalTempVariation,
        double basePh,
        double phVariability
) {
    @java.lang.Override
    public long seed() {
        return seed;
    }

    @java.lang.Override
    public double totalLength() {
        return totalLength;
    }

    @java.lang.Override
    public double spatialResolution() {
        return spatialResolution;
    }

    @java.lang.Override
    public double initialElevation() {
        return initialElevation;
    }

    @java.lang.Override
    public double averageSlope() {
        return averageSlope;
    }

    @java.lang.Override
    public double slopeVariability() {
        return slopeVariability;
    }

    @java.lang.Override
    public double baseWidth() {
        return baseWidth;
    }

    @java.lang.Override
    public double widthVariability() {
        return widthVariability;
    }

    @java.lang.Override
    public double baseSideSlope() {
        return baseSideSlope;
    }

    @java.lang.Override
    public double sideSlopeVariability() {
        return sideSlopeVariability;
    }

    @java.lang.Override
    public double baseManning() {
        return baseManning;
    }

    @java.lang.Override
    public double manningVariability() {
        return manningVariability;
    }

    @java.lang.Override
    public double baseDecayRateAt20C() {
        return baseDecayRateAt20C;
    }

    @java.lang.Override
    public double decayRateVariability() {
        return decayRateVariability;
    }

    @java.lang.Override
    public double baseTemperature() {
        return baseTemperature;
    }

    @java.lang.Override
    public double dailyTempVariation() {
        return dailyTempVariation;
    }

    @java.lang.Override
    public double seasonalTempVariation() {
        return seasonalTempVariation;
    }

    @java.lang.Override
    public double basePh() {
        return basePh;
    }

    @java.lang.Override
    public double phVariability() {
        return phVariability;
    }
}