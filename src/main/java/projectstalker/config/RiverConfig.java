package projectstalker.config;


/**
 * Un objeto de valor inmutable para contener todos los parámetros de configuración
 * necesarios para la generación procedural de un río.
 * <p>
 * Esta clase agrupa todas las variables de alto nivel que definen la morfología,
 * hidrología y las propiedades físico-químicas básicas del río a simular.
 *
 * @param seed Semilla para la generación de ruido, para resultados reproducibles.
 * @param noiseFrequency Controla el nivel de detalle y la escala de las características del ruido
 *                       - Una baja frecuencia equivale a colinas grandes, suaves y muy espaciadas.
 *                       Las transiciones de un valor a otro son lentas y graduales
 *                       - Una alta frecuencia equivale a muchas colinas pequeñas, juntas y con pendientes pronunciadas.
 *                       Las transiciones son abruptas
 *
 * @param totalLength Longitud total del río en metros.
 * @param spatialResolution Resolución espacial (dx) en metros.
 * @param initialElevation Altitud inicial del río en metros.
 * @param concavityFactor Controla la variación de la pendiente del río en función de la posición
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
 * @param averageAnnualTemperature Media de temperatura anual
 */
public record RiverConfig(
        long seed,
        float noiseFrequency,
        float detailNoiseFrequency,
        float zoneNoiseFrequency,
        double totalLength,
        double spatialResolution,
        double initialElevation,
        double concavityFactor,
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
        double averageAnnualTemperature,
        double basePh,
        double phVariability
) {
    // Atención, el record ya escribe los getters, el constructor y hace todas las variables private final
}