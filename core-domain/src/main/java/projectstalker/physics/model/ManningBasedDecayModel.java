package projectstalker.physics.model;

import projectstalker.config.RiverConfig;

/**
 * Estrategia Espacial para el Coeficiente de Reacción (Decay).
 * <p>
 * Basa la capacidad de autodepuración del río en su turbulencia física (Rugosidad).
 * <ul>
 * <li><b>Manning Alto (Rápido/Rugoso):</b> Mayor turbulencia y reaireación.
 * Aumenta la eficiencia bacteriana (Decay sube).</li>
 * <li><b>Manning Bajo (Lento/Liso):</b> Menor mezcla de oxígeno.
 * La descomposición se ralentiza (Decay baja).</li>
 * </ul>
 */
public class ManningBasedDecayModel implements SpatialModel {

    /**
     * @param cellIndex   Índice de la celda.
     * @param config      Configuración global.
     * @param localManning Valor del Coeficiente de Manning local (Driver).
     * @param localNoise  Ruido de detalle (Biomasa local/Biofilm).
     */
    @Override
    public float calculate(int cellIndex, RiverConfig config, double localManning, double localNoise) {
        double baseDecay = config.baseDecayRateAt20C();
        double baseManning = config.baseManning();

        // 1. Factor de Turbulencia (Relativo al Manning medio del río)
        // Si el tramo es más rugoso que la media -> Factor > 1.0
        double turbulenceRatio = localManning / Math.max(EPSILON, baseManning);

        // Aplicamos sensibilidad
        double turbulenceFactor = Math.pow(turbulenceRatio, config.decayTurbulenceSensitivity());

        // Acotamos: La turbulencia puede duplicar el decay, pero no hacerlo infinito.
        // Rango: 0.5x (aguas estancadas) a 2.0x (rápidos muy oxigenados)
        turbulenceFactor = Math.max(0.5, Math.min(2.0, turbulenceFactor));

        // 2. Variabilidad Biológica Local (Ruido)
        // Simula parches de vegetación o colonias bacterianas concentradas.
        double bioVariability = localNoise * config.decayRateVariability();

        // 3. Cálculo Final
        double finalDecay = (baseDecay * turbulenceFactor) + bioVariability;

        // El decay nunca puede ser negativo.
        return (float) Math.max(0.001, finalDecay);
    }
}