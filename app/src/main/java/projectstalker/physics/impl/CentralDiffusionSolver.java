package projectstalker.physics.impl;

import lombok.Builder;
import lombok.With;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.physics.i.IDiffusionSolver;

@Builder
@With
public class CentralDiffusionSolver implements IDiffusionSolver {

    private final double minDiffusionCoefficient;

    public CentralDiffusionSolver() {
        this(1e-6); // Valor por defecto para evitar división por cero o estancamiento
    }

    public CentralDiffusionSolver(double minDiffusionCoefficient) {
        this.minDiffusionCoefficient = minDiffusionCoefficient;
    }

    @Override
    public String getName() {
        return "CentralDiff_2ndOrder";
    }

    @Override
    public float[] solveDiffusion(float[] concentration, float[] velocity, float[] depth, RiverGeometry geometry, float dt) {
        int n = concentration.length;
        float[] result = new float[n];
        double dx = geometry.getSpatialResolution();
        double dx2 = dx * dx;

        for (int i = 1; i < n - 1; i++) {
            double alpha = geometry.getDispersionAlphaAt(i);
            // Asumimos velocidad positiva (RiverState actual) pero Math.abs es seguro
            double u = velocity[i];
            double h = depth[i];

            double DL = alpha * u * h;
            if (DL < this.minDiffusionCoefficient) DL = this.minDiffusionCoefficient;

            double r = (DL * dt) / dx2;

            // Advertencia silenciosa: Si r > 0.5, este esquema es inestable.
            // El Orquestador (SplitOperator) es responsable de darnos un dt pequeño.

            double term = concentration[i + 1] - 2.0 * concentration[i] + concentration[i - 1];
            result[i] = (float) (concentration[i] + r * term);
        }

        // Condiciones de Frontera (Hardcoded como Neumann cero por ahora, lo estándar en ríos)
        result[0] = result[1];
        result[n - 1] = result[n - 2];

        return result;
    }
}