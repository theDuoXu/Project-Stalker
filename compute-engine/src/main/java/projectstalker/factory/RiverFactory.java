package projectstalker.factory;

import projectstalker.config.RiverConfig;
import projectstalker.domain.river.InitialRiver;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;
import projectstalker.physics.solver.ManningEquationSolver;

public class RiverFactory {

    private static final float MIN_DEPTH = 0.001f; // 1mm de agua mínima para evitar div/0
    private static final int MAX_NEWTON_ITERATIONS = 10;
    private static final float TOLERANCE = 1e-4f;

    // Dependencia opcional, solo si necesitas generar geometría desde cero
    private final RiverGeometryFactory geometryFactory;

    public RiverFactory(RiverGeometryFactory geometryFactory) {
        this.geometryFactory = geometryFactory;
    }

    /**
     * Crea un río en ESTADO ESTACIONARIO (Steady State) de forma analítica.
     * Delega el cálculo físico a {@link ManningEquationSolver}.
     */
    public static RiverState createSteadyState(RiverGeometry geometry, float baseDischarge) {
        int cells = geometry.getCellCount();

        float[] h = new float[cells];
        float[] v = new float[cells];
        float[] zero = new float[cells];

        // Calcular pendientes locales
        float[] slopes = calculateSlopes(geometry);
        float[] manning = geometry.getManningCoefficient();
        float[] widths = geometry.getBottomWidth();
        float[] sideSlopes = geometry.getSideSlope();

        for (int i = 0; i < cells; i++) {
            // Delegación limpia a la librería de física
            h[i] = ManningEquationSolver.calculateNormalDepth(
                    baseDischarge,
                    slopes[i],
                    manning[i],
                    widths[i],
                    sideSlopes[i]
            );

            v[i] = ManningEquationSolver.calculateVelocity(
                    baseDischarge,
                    h[i],
                    widths[i],
                    sideSlopes[i]
            );
        }

        return RiverState.builder()
                .waterDepth(h)
                .velocity(v)
                .temperature(zero)
                .ph(zero)
                .contaminantConcentration(zero)
                .build();
    }

    /**
     * Resuelve la ecuación de Manning para H (Profundidad) usando Newton-Raphson.
     * f(H) = Q_calc(H) - Q_target = 0
     */
    private static float solveNormalDepth(float Q_target, float slope, float n, float b, float m) {
        float sqrtSlope = (float) Math.sqrt(Math.max(slope, 1e-6f)); // Slope positivo forzado
        float H = 1.0f; // Semilla inicial (1 metro)

        for (int k = 0; k < MAX_NEWTON_ITERATIONS; k++) {
            float A = (b + m * H) * H;
            float P = b + 2 * H * (float) Math.sqrt(1 + m * m);
            float R = A / P;

            // Manning: Q = (1/n) * A * R^(2/3) * S^(1/2)
            float Q_calc = (1.0f / n) * A * (float) Math.pow(R, 2.0/3.0) * sqrtSlope;

            float f = Q_calc - Q_target;
            if (Math.abs(f) < TOLERANCE) return H;

            // Derivada dQ/dH (Aproximada o Analítica)
            // Usamos aproximación numérica simple para robustez o analítica si prefieres:
            // dQ/dH ≈ Q * ( (5/3)*(T/A) - (2/3)*(dP_dH/P) )
            float T = b + 2 * m * H; // Top width
            float dP_dH = 2 * (float) Math.sqrt(1 + m * m);
            float df = Q_calc * ((5.0f/3.0f)*(T/A) - (2.0f/3.0f)*(dP_dH/P));

            // Paso Newton
            if (Math.abs(df) < 1e-9) break; // Evitar explosión
            H = H - (f / df);

            if (H < MIN_DEPTH) H = MIN_DEPTH; // Clamp
        }
        return H;
    }

    // Helper para calcular pendientes del terreno
    private static float[] calculateSlopes(RiverGeometry geo) {
        int n = geo.getCellCount();
        float[] s = new float[n];
        float[] elev = geo.getElevationProfile();
        float dx = geo.getSpatialResolution();

        for (int i = 0; i < n - 1; i++) {
            // Pendiente descendente positiva
            s[i] = (elev[i] - elev[i+1]) / dx;
            // Evitar pendiente negativa o cero (agua estancada no fluye en Manning simple)
            if (s[i] < 1e-5f) s[i] = 1e-5f;
        }
        s[n-1] = s[n-2]; // Extrapolar última celda
        return s;
    }

    // Método de compatibilidad si aún necesitas crear InitialRiver
    @Deprecated
    public InitialRiver createStableRiver(RiverConfig config, float initialDischarge) {
        RiverGeometry geom = geometryFactory.createRealisticRiver(config);
        RiverState state = createSteadyState(geom, initialDischarge);
        return new InitialRiver(geom, state);
    }
}