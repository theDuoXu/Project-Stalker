package projectstalker.physics.model;

import projectstalker.config.RiverConfig;

/**
 * Modelo de Ancho basado en Pendiente (Geometría Hidráulica).
 * <p>
 * Implementa la correlación física inversa entre pendiente y ancho del cauce:
 * <ul>
 * <li><b>Zonas Abruptas (Alta Pendiente):</b> El flujo se acelera y el cauce se estrecha (efecto cañón/incisión).</li>
 * <li><b>Zonas Planas (Baja Pendiente):</b> El flujo se desacelera y el cauce tiende a ensancharse y divagar.</li>
 * </ul>
 */
public class SlopeBasedWidthModel implements SpatialModel {

    @Override
    public float calculate(int cellIndex, RiverConfig config, double localSlope, double noiseFactor) {

        // 1. Obtener valores base
        float baseWidth = config.baseWidth();
        float avgSlope = config.averageSlope();
        float slopeSensitivityExponent = config.slopeSensitivityExponent();
        // 2. Evitar división por cero o pendientes negativas en el cálculo físico
        // Usamos un valor mínimo seguro (epsilon) para la pendiente local.
        double safeLocalSlope = Math.max(EPSILON, localSlope);

        // 3. Calcular el Ratio de Pendiente (Local vs Media)
        // Si localSlope > avgSlope (más empinado) -> ratio > 1.0
        // Si localSlope < avgSlope (más plano) -> ratio < 1.0
        double slopeRatio = safeLocalSlope / avgSlope;

        // 4. Aplicar la Ley de Potencia Inversa (Geometría Hidráulica)
        // WidthFactor = 1 / (SlopeRatio ^ k)
        // Si es más empinado, el factor será < 1 (se estrecha).
        // Si es más plano, el factor será > 1 (se ensancha).
        double hydraulicFactor = 1.0 / Math.pow(slopeRatio, slopeSensitivityExponent);

        // 5. Acotar el factor hidráulico para evitar extremos irreales
        // Limitamos el ensanchamiento a 3x y el estrechamiento a 0.3x
        hydraulicFactor = Math.max(0.3, Math.min(3.0, hydraulicFactor));

        // 6. Integrar la variabilidad estocástica (Ruido)
        // El ruido afecta porcentualmente al ancho resultante (ej: +/- 20%)
        // widthVariability en config se trata aquí como amplitud relativa.
        double variability = config.widthVariability();
        // Aquí asumimos el uso aditivo del ruido modulado.
        double noiseComponent = noiseFactor * variability;

        // 7. Resultado Final
        double finalWidth = (baseWidth * hydraulicFactor) + noiseComponent;

        // Aseguramos que el río nunca tenga ancho negativo o cero
        return (float) Math.max(1.0, finalWidth);
    }
}