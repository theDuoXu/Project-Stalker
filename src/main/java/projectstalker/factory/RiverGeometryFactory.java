package projectstalker.factory;

import projectstalker.config.RiverConfig;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.utils.FastNoiseLite;

/**
 * Fábrica responsable de la creación procedural de instancias de {@link RiverGeometry}.
 * <p>
 * Encapsula la lógica compleja de generación de perfiles de río realistas,
 * utilizando ruido Perlin para la variabilidad natural y asegurando que el
 * objeto final sea siempre físicamente consistente y válido.
 *
 * @author Duo Xu
 * @version 1.1
 * @since 2025-10-13
 */
public class RiverGeometryFactory {

    private static final float NOISE_FREQUENCY = 0.05f; // Controla la "escala" de las variaciones.

    /**
     * Crea una instancia de RiverGeometry con características realistas
     * generadas proceduralmente a partir de una configuración dada.
     *
     * @param config El objeto de configuración que define las propiedades del río.
     * @return Un objeto RiverGeometry inmutable y físicamente consistente.
     */
    public RiverGeometry createRealisticRiver(RiverConfig config) {
        // 1. Inicialización
        final int cellCount = (int) Math.round(config.totalLength() / config.spatialResolution());

        // Inicializamos el generador de ruido con la semilla para reproducibilidad
        final FastNoiseLite noise = new FastNoiseLite((int) config.seed());
        noise.SetNoiseType(FastNoiseLite.NoiseType.Perlin);
        noise.SetFrequency(NOISE_FREQUENCY);

        // 2. Creación de los arrays de atributos
        double[] elevationProfile = new double[cellCount];
        double[] bottomWidth = new double[cellCount];
        double[] sideSlope = new double[cellCount];
        double[] manningCoefficient = new double[cellCount];
        double[] baseDecayCoefficientAt20C = new double[cellCount];
        double[] phProfile = new double[cellCount];

        // 3. Generación procedural de los perfiles celda por celda
        elevationProfile[0] = config.initialElevation();

        for (int i = 0; i < cellCount; i++) {
            // Obtenemos un valor de ruido suave para la posición actual
            double currentNoise = noise.GetNoise(i, 0);

            // --- Generar Elevación (Paso Crítico para la validación) ---
            if (i > 0) {
                double baseDrop = config.averageSlope() * config.spatialResolution();
                double noiseEffectOnDrop = currentNoise * config.slopeVariability();

                // Aseguramos que la caída siempre sea positiva (o cero) para que el perfil
                // sea monotónicamente no creciente, cumpliendo así la validación.
                double totalDrop = Math.max(0, baseDrop + noiseEffectOnDrop);
                elevationProfile[i] = elevationProfile[i - 1] - totalDrop;
            }

            // --- Generar Ancho del Fondo ---
            double widthValue = config.baseWidth() + currentNoise * config.widthVariability();
            bottomWidth[i] = Math.max(0.1, widthValue); // Evitar ancho cero o negativo

            // --- Generar Pendiente de Taludes ---
            double sideSlopeValue = config.baseSideSlope() + currentNoise * config.sideSlopeVariability();
            sideSlope[i] = Math.max(0, sideSlopeValue); // La pendiente no puede ser negativa

            // --- Generar Coeficiente de Manning ---
            double manningValue = config.baseManning() + currentNoise * config.manningVariability();
            manningCoefficient[i] = Math.max(0.01, manningValue); // Debe ser siempre positivo

            // --- Generar Coeficiente de Reacción Base ---
            double decayValue = config.baseDecayRateAt20C() + currentNoise * config.decayRateVariability();
            baseDecayCoefficientAt20C[i] = Math.max(0.0, decayValue); // La reacción no puede ser negativa

            // --- Generar Perfil de pH ---
            double phValue = config.basePh() + currentNoise * config.phVariability();
            phProfile[i] = Math.max(6.0, Math.min(9.0, phValue)); // Mantener en un rango plausible [6.0, 9.0]
        }

        // 4. Instanciar y devolver el objeto RiverGeometry final y validado
        return new RiverGeometry(
                cellCount,
                config.spatialResolution(),
                elevationProfile,
                bottomWidth,
                sideSlope,
                manningCoefficient,
                baseDecayCoefficientAt20C,
                phProfile
        );
    }
}