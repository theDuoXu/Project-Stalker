package projectstalker.physics.impl;

import projectstalker.domain.river.RiverGeometry;
import projectstalker.physics.i.IReactionSolver;

/**
 * Implementación de cinética química de primer orden (Decaimiento exponencial).
 * Útil para DBO (Demanda Biológica de Oxígeno), patógenos o isótopos radiactivos.
 */
public class FirstOrderReactionSolver implements IReactionSolver {

    private static final double THETA = 1.047;
    private static final double REF_TEMP = 20.0;

    @Override
    public String getName() {
        return "FirstOrderDecay";
    }

    @Override
    public String getDescription() {
        return "Cinética de 1er orden corregida por temperatura: k(T) = k20 * theta^(T-20)";
    }

    @Override
    public float[] solveReaction(float[] concentration, float[] temperature, RiverGeometry geometry, float dt) {
        int n = concentration.length;
        float[] result = new float[n];

        for (int i = 0; i < n; i++) {
            // 1. Obtener k base a 20°C
            double k20 = geometry.getBaseDecayAt(i);

            // 2. Obtener temperatura actual (si el array es nulo o vacío, asumimos 20°C)
            double T = (temperature != null && i < temperature.length) ? temperature[i] : REF_TEMP;

            // 3. Aplicar corrección de Arrhenius
            // k_real = k20 * theta^(T - 20)
            double kReal = k20 * Math.pow(THETA, T - REF_TEMP);

            // 4. Decaimiento
            double decayFactor = Math.exp(-kReal * dt);
            result[i] = (float) (concentration[i] * decayFactor);
        }

        return result;
    }
}