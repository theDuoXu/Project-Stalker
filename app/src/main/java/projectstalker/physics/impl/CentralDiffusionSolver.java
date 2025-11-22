package projectstalker.physics.impl;

import projectstalker.domain.river.RiverGeometry;
import projectstalker.physics.i.IDiffusionSolver;

/**
 * Solver de Difusión/Dispersión utilizando Diferencias Centrales de 2º Orden.
 * Calcula el coeficiente de dispersión longitudinal (D_L) dinámicamente.
 */
public class CentralDiffusionSolver implements IDiffusionSolver {

    @Override
    public String getName() {
        return "CentralDiff_2ndOrder";
    }

    @Override
    public float[] solveDiffusion(float[] concentration, float[] velocity, float[] depth, RiverGeometry geometry, float dt) {
        int n = concentration.length;
        float[] result = new float[n];
        double dx = geometry.getSpatial_resolution();
        double dx2 = dx * dx; // dx al cuadrado pre-calculado

        // 1. Celdas Interiores (1 a N-2)
        for (int i = 1; i < n - 1; i++) {
            // Calcular Coeficiente de Dispersión Local: D_L = alpha * |u| * H
            // alpha viene de la geometría (factor de dispersión de Taylor)
            double alpha = geometry.getDispersionAlphaAt(i);
            double u = Math.abs(velocity[i]);
            double h = depth[i];

            double DL = alpha * u * h;

            // Evitar difusión nula numérica si el agua está quieta (siempre hay algo de difusión molecular)
            if (DL < 1e-6) DL = 1e-6;

            // Número de difusión (r). Para estabilidad explícita, r <= 0.5
            // (El control de estabilidad lo hará el orquestador reduciendo dt)
            double r = (DL * dt) / dx2;

            // Esquema explícito de diferencias centrales:
            // C_new = C_i + r * (C_next - 2*C_i + C_prev)
            double term = concentration[i + 1] - 2.0 * concentration[i] + concentration[i - 1];
            result[i] = (float) (concentration[i] + r * term);
        }

        // 2. Condiciones de Frontera (Boundary Conditions)

        // Aguas Arriba (i=0): Dirichlet o Neumann cero (Asumimos no flujo difusivo hacia atrás por ahora)
        // Copiamos el valor calculado en i=1 para simular pendiente cero (Neumann)
        result[0] = result[1];

        // Aguas Abajo (i=N-1): Frontera Transmisiva (Neumann, pendiente cero)
        // El contaminante sale libremente, no rebota.
        result[n - 1] = result[n - 2];

        return result;
    }
}