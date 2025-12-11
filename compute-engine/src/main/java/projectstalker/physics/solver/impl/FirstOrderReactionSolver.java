package projectstalker.physics.solver.impl;

import lombok.Builder;
import lombok.With;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.physics.solver.ReactionSolver;
import java.util.stream.IntStream;

@Builder
@With
public class FirstOrderReactionSolver implements ReactionSolver {

    // Configuración inmutable
    private final double thetaArrhenius;
    private final double referenceTemperature;
    private final boolean useParallelExecution;

    /**
     * Constructor por defecto (Valores estándar para calidad de aguas).
     */
    public FirstOrderReactionSolver() {
        this(1.047, 20.0, false);
    }

    /**
     * Constructor configurable.
     * @param thetaArrhenius Coeficiente de temperatura (típicamente 1.0 - 1.1).
     * @param referenceTemperature Temperatura de referencia donde k es conocida (típicamente 20°C).
     * @param useParallelExecution Si es true, usa ForkJoinPool para arrays muy grandes.
     */
    public FirstOrderReactionSolver(double thetaArrhenius, double referenceTemperature, boolean useParallelExecution) {
        this.thetaArrhenius = thetaArrhenius;
        this.referenceTemperature = referenceTemperature;
        this.useParallelExecution = useParallelExecution;
    }

    @Override
    public String getName() {
        return "Reaction_1stOrder_Arrhenius";
    }

    @Override
    public float[] solveReaction(float[] concentration, float[] temperature, RiverGeometry geometry, float dt) {
        int n = concentration.length;
        float[] result = new float[n];

        // Usar Parallel Streams correctamente si se solicita
        if (this.useParallelExecution && n > 10_000) {
            IntStream.range(0, n).parallel().forEach(i -> computeCell(i, concentration, temperature, geometry, dt, result));
        } else {
            // Bucle simple para tamaños normales (evita overhead de hilos)
            for (int i = 0; i < n; i++) {
                computeCell(i, concentration, temperature, geometry, dt, result);
            }
        }

        return result;
    }

    // Lógica extraída para ser usada tanto en secuencial como paralelo
    private void computeCell(int i, float[] c, float[] t, RiverGeometry geo, float dt, float[] res) {
        double k20 = geo.getBaseDecayAt(i);

        // Si no hay datos de temperatura, usamos la de referencia (sin corrección)
        double temp = (t != null && i < t.length) ? t[i] : this.referenceTemperature;

        double kReal = k20 * Math.pow(this.thetaArrhenius, temp - this.referenceTemperature);
        double decayFactor = Math.exp(-kReal * dt);

        res[i] = (float) (c[i] * decayFactor);
    }
}