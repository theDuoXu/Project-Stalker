package projectstalker.physics.model;

import projectstalker.config.RiverConfig;

/**
 * Estrategia de Dispersión Longitudinal (Alpha).
 * <p>
 * Modela el coeficiente de dispersión basándose en la Hidráulica:
 * <ul>
 * <li><b>Correlación Directa con Manning:</b> Un lecho rugoso (alto Manning) genera
 * zonas muertas y remolinos que atrapan y sueltan el soluto, aumentando la dispersión.</li>
 * <li><b>Fórmula:</b> Alpha escala proporcionalmente al ratio de rugosidad local vs base.</li>
 * </ul>
 */
public class ManningBasedDispersionModel implements SpatialModel {

    @Override
    public float calculate(int cellIndex, RiverConfig config, double localManning, double localNoise) {
        double baseAlpha = config.baseDispersionAlpha();
        double baseManning = config.baseManning();

        // 1. Factor Hidráulico (Ratio de Rugosidad)
        // Evitamos división por cero con un epsilon seguro
        double roughnessRatio = localManning / Math.max(EPSILON, baseManning);

        // 2. Variabilidad Estocástica
        double noiseComponent = localNoise * config.alphaVariability();

        // 3. Cálculo Final
        // Alpha ~ BaseAlpha * (n_local / n_base) + Ruido
        double finalAlpha = (baseAlpha * roughnessRatio) + noiseComponent;

        // 4. Clamping
        // La dispersión siempre es positiva. 0.1 es un mínimo físico razonable.
        return (float) Math.max(0.1, finalAlpha);
    }
}