package projectstalker.physics.solver;

import projectstalker.domain.river.RiverGeometry;

/**
 * Clase que encapsula la lógica para resolver la ecuación de Manning.
 * Utiliza el method de Newton-Raphson para encontrar la profundidad (H)
 * a partir de un caudal (Q) dado.
 */
public final class ManningEquationSolver {

    private static final int MAX_ITERATIONS = 20;
    private static final double TOLERANCE = 1e-6; // Tolerancia para convergencia en metros

    /**
     * Prohibido construir esta clase utilidad
     */
    private ManningEquationSolver() {
    }

    /**
     * Encuentra la profundidad del agua (H) que corresponde a un caudal dado.
     * Esta clase es Thread safe
     *
     * @param targetDischarge   El caudal objetivo que la celda debe evacuar (m³/s).
     * @param initialDepthGuess Una estimación inicial para la profundidad (m).
     * @param cellIndex         El índice de la celda del río.
     * @param geometry          La geometría del río.
     * @return La profundidad calculada (m).
     */

    public static double findDepth(double targetDischarge, double initialDepthGuess, int cellIndex, RiverGeometry geometry) {

        // Extraer parámetros geométricos una sola vez
        final double b = geometry.getWidthAt(cellIndex);
        final double m = geometry.getSideSlopeAt(cellIndex);
        final double n = geometry.getManningAt(cellIndex);
        double S = geometry.getBedSlopeAt(cellIndex);

        // Evitar pendiente cero o negativa que impide el cálculo.
        if (S <= 1e-7) {
            S = 1e-7;
        }
        final double sqrtSlope = Math.sqrt(S);
        final double pythagoras = Math.sqrt(1.0 + m * m);

        double H = (initialDepthGuess <= 0) ? 0.1 : initialDepthGuess; // Empezar con un valor pequeño si la estimación es cero

        for (int i = 0; i < MAX_ITERATIONS; i++) {
            // Función f(H) = Q_calculado(H) - Q_objetivo
            double calculatedDischarge = calculateQ(H, b, m, n, sqrtSlope,  pythagoras);
            double f_H = calculatedDischarge - targetDischarge;

            // Si el error es suficientemente pequeño, hemos convergido.
            if (Math.abs(f_H) < TOLERANCE) {
                return H;
            }

            // Derivada f'(H) = dQ/dH
            double dQ_dH = calculate_dQ_dH(H, b, m, calculatedDischarge, pythagoras);

            // Evitar división por cero o un gradiente que no permita avanzar.
            if (Math.abs(dQ_dH) < 1e-6) {
                break; // Salir si la derivada es plana, el method no convergerá
            }

            // Fórmula de Newton-Raphson
            double h_next = H - f_H / dQ_dH;

            // Comprobar convergencia por cambio en H
            if (Math.abs(h_next - H) < TOLERANCE) {
                return Math.max(0, h_next);
            }

            // Actualizar H para la siguiente iteración, asegurando que no sea negativo.
            H = Math.max(0.001, h_next); // Evitar que H sea exactamente cero para prevenir singularidades
        }

        // Si no converge, devuelve la última mejor estimación.
        return H;
    }

    /**
     * Calcula el caudal (Q) usando la ecuación de Manning para un canal trapezoidal.
     */
    private static double calculateQ(double H, double b, double m, double n, double sqrtSlope, double pythagoras) {
        double A = (b + m * H) * H;
        if (A <= 0) return 0.0;

        // Usamos el pre-cálculo de Pitágoras
        double P = b + 2.0 * H * pythagoras;
        if (P <= 1e-9) return 0.0;

        double R = A / P;
        // Usamos sqrtSlope pre-calculado
        return (1.0 / n) * A * Math.pow(R, 2.0 / 3.0) * sqrtSlope;
    }

    /**
     * Calcula la derivada del caudal respecto a la profundidad (dQ/dH) analíticamente.
     */
    private static double calculate_dQ_dH(double H, double b, double m, double currentQ, double pythagoras) {
        if (H <= 1e-9) return Double.POSITIVE_INFINITY;

        // Derivadas optimizadas
        double topWidth = b + 2.0 * m * H;
        double A = (b + m * H) * H;
        double P = b + 2.0 * H * pythagoras;

        double term_A_derivative = (5.0 * topWidth) / (3.0 * A);
        double term_P_derivative = (4.0 * pythagoras) / (3.0 * P);

        // Factorizamos: Q * ( (5/3)*T/A - (2/3)*dP/P )
        return currentQ * (term_A_derivative - term_P_derivative);
    }
}