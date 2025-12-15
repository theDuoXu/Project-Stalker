package projectstalker.physics.model;

import projectstalker.config.RiverConfig;

/**
 * Modelo de Taludes (Side Slope) basado en Material (Ángulo de Reposo).
 * <p>
 * Vincula la rugosidad (Manning) con la estabilidad del talud:
 * <ul>
 * <li><b>Manning Bajo (Arena):</b> Usamos <code>config.baseSideSlope()</code>. Talud suave (z alto).</li>
 * <li><b>Manning Alto (Roca):</b> Usamos <code>base - variability</code>. Talud empinado (z bajo).</li>
 * </ul>
 */
public class MaterialBasedSideSlopeModel implements SpatialModel {

    // Referencias de Manning para la interpolación (Física de materiales)
    private static final double MANNING_SAND_REF = 0.025; // n para arena limpia/tierra
    private static final double MANNING_ROCK_REF = 0.055; // n para roca/montaña

    @Override
    public float calculate(int cellIndex, RiverConfig config, double localManning, double unusedNoise) {

        // 1. Definir el Rango de Taludes desde la Configuración
        // z = Distancia Horizontal / Altura Vertical

        // El talud Base es el más suave (Arena), ej: 4.0
        double zSand = config.baseSideSlope();

        // El talud Roca es el base menos la variabilidad, ej: 4.0 - 3.5 = 0.5 (Vertical)
        // Clampeamos a 0.1 para no tener paredes negativas o división por cero infinita
        double z = getZ(config, localManning, zSand);

        return (float) z;
    }

    private static double getZ(RiverConfig config, double localManning, double zSand) {
        double zRock = Math.max(0.1, zSand - config.sideSlopeVariability());

        // 2. Normalizar el Manning local
        // Determinamos dónde cae el manning actual entre arena y roca
        double n = Math.max(MANNING_SAND_REF, Math.min(MANNING_ROCK_REF, localManning));

        // Ratio 0.0 = Arena, 1.0 = Roca
        double hardnessRatio = (n - MANNING_SAND_REF) / (MANNING_ROCK_REF - MANNING_SAND_REF);

        // 3. Interpolar el talud z
        // Si ratio es 0 (Arena) -> devolvemos zSand (4.0)
        // Si ratio es 1 (Roca)  -> devolvemos zRock (0.5)
        double z = zSand - (hardnessRatio * (zSand - zRock));
        return z;
    }
}