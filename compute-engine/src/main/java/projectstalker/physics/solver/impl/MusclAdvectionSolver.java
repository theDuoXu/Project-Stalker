package projectstalker.physics.solver.impl;

import lombok.Builder;
import lombok.With;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.physics.solver.AdvectionSolver;

@Builder
@With
public class MusclAdvectionSolver implements AdvectionSolver {

    public enum Limiter { MINMOD, SUPERBEE, VAN_LEER }

    private final float upstreamConcentration;
    private final Limiter limiterType;

    /**
     * Constructor por defecto. Asume entrada limpia y limitador MinMod.
     */
    public MusclAdvectionSolver() {
        this(0.0f, Limiter.MINMOD);
    }

    /**
     * Constructor configurado.
     * @param upstreamConcentration Concentración constante que entra por la frontera (mg/L).
     * @param limiter Tipo de limitador de flujo para TVD.
     */
    public MusclAdvectionSolver(float upstreamConcentration, Limiter limiter) {
        this.upstreamConcentration = upstreamConcentration;
        this.limiterType = limiter;
    }

    @Override
    public String getName() {
        return "MUSCL_" + limiterType.name();
    }

    @Override
    public float[] solveAdvection(float[] concentration, float[] velocity, float[] area, RiverGeometry geometry, float dt) {
        int n = concentration.length;
        float[] newConcentration = new float[n];
        double dx = geometry.getSpatialResolution();
        float[] flux = new float[n + 1];

        // 1. CALCULAR FLUJOS
        for (int i = 0; i < n - 1; i++) {
            if (i == 0 || i >= n - 2) {
                // Borde: Upwind (1er orden) para estabilidad en extremos
                double Q = velocity[i] * area[i];
                flux[i] = (float) (Q * concentration[i]);
            } else {
                // Interior: MUSCL (2do orden)
                flux[i] = calculateMusclFlux(concentration, velocity, area, i);
            }
        }

        // --- CONDICIONES DE FRONTERA CONFIGURABLES ---

        // Frontera Aguas Arriba (Inlet): Dirichlet
        // Calculamos el caudal entrante (Q_in) basándonos en la primera celda
        double Q_in = velocity[0] * area[0];
        // Flux = Q * Concentración configurada en el constructor
        float fluxIn = (float) (Q_in * this.upstreamConcentration);

        // Frontera Aguas Abajo (Outlet): Transmisiva (Neumann dC/dx = 0)
        double Q_out = velocity[n - 1] * area[n - 1];
        flux[n - 1] = (float) (Q_out * concentration[n - 1]);

        // 2. ACTUALIZAR ESTADO
        for (int i = 0; i < n; i++) {
            float fluxOutRight = flux[i];
            float fluxInLeft = (i == 0) ? fluxIn : flux[i - 1];

            double volume = area[i] * dx;

            if (volume > 1e-9) {
                double change = (dt / volume) * (fluxInLeft - fluxOutRight);
                newConcentration[i] = (float) (concentration[i] + change);
            } else {
                newConcentration[i] = 0.0f;
            }

            if (newConcentration[i] < 0) newConcentration[i] = 0.0f;
        }

        return newConcentration;
    }

    private float calculateMusclFlux(float[] c, float[] u, float[] A, int i) {
        int prev = i - 1; int curr = i; int next = i + 1;

        float slope_left = c[curr] - c[prev];
        float slope_right = c[next] - c[curr];

        // Usamos el limitador configurado
        float limitedSlope = applyLimiter(slope_left, slope_right);

        float c_left_at_face = c[curr] + 0.5f * limitedSlope;
        double Q = u[curr] * A[curr];

        return (float) (Q * c_left_at_face);
    }

    private float applyLimiter(float a, float b) {
        // Si tienen signos opuestos, es un pico -> pendiente 0 para todos los limitadores TVD
        if (a * b <= 0) return 0.0f;

        switch (this.limiterType) {
            case SUPERBEE:
                // Superbee: max(0, min(2a, b), min(a, 2b))
                // Es más "agresivo", mantiene los bordes de la mancha más verticales.
                float absA = Math.abs(a);
                float absB = Math.abs(b);
                if (a > 0) {
                    return Math.max(0, Math.max(Math.min(2*a, b), Math.min(a, 2*b)));
                } else {
                    return -Math.max(0, Math.max(Math.min(2*absA, absB), Math.min(absA, 2*absB)));
                }
            case VAN_LEER:
                // Van Leer: (r + |r|) / (1 + |r|) ... implementación armónica
                return (2.0f * a * b) / (a + b);
            case MINMOD:
            default:
                // MinMod: El más seguro y difusivo.
                return (Math.abs(a) < Math.abs(b)) ? a : b;
        }
    }
}