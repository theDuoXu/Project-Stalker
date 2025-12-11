package projectstalker.physics.solver.impl;

import projectstalker.domain.river.RiverGeometry;

/**
 * Biblioteca estática de algoritmos para resolver la Ecuación de Manning.
 * <p>
 * Contiene métodos analíticos y numéricos (Newton-Raphson) para determinar
 * profundidad normal, velocidad y caudal en canales abiertos.
 * <p>
 * Stateless y Thread-Safe.
 */
public final class ManningEquationSolver {

    private static final int MAX_ITERATIONS = 20;
    private static final double TOLERANCE = 1e-6; // Tolerancia convergencia

    private ManningEquationSolver() {}

    /**
     * Versión "Convenience" que extrae los datos de la geometría.
     * Mantenida para compatibilidad con tu código existente.
     */
    public static float findDepth(float targetDischarge, float initialDepthGuess, int cellIndex, RiverGeometry geometry) {
        // Extraer parámetros
        float b = geometry.getWidthAt(cellIndex);
        float m = geometry.getSideSlopeAt(cellIndex);
        float n = geometry.getManningAt(cellIndex);
        float S = geometry.getBedSlopeAt(cellIndex);

        return calculateNormalDepth(targetDischarge, S, n, b, m, initialDepthGuess);
    }

    /**
     * Versión "Primitiva" (Core Matemático).
     * Usada por RiverFactory para calcular el estado inicial sin instanciar simuladores.
     *
     * @param targetDischarge Caudal objetivo (m³/s)
     * @param slope           Pendiente del fondo (m/m)
     * @param n               Coeficiente de Manning
     * @param b               Ancho del fondo (m)
     * @param m               Pendiente lateral (z)
     * @param initialGuess    (Opcional) Semilla inicial. Si es <=0 se usa 1.0m
     */
    public static float calculateNormalDepth(float targetDischarge, float slope, float n, float b, float m, float initialGuess) {
        // Validaciones físicas
        if (slope <= 1e-7) slope = 1e-7f;
        if (Math.abs(targetDischarge) < 1e-9) return 0f;

        final double sqrtSlope = Math.sqrt(slope);
        final double pythagoras = Math.sqrt(1.0 + m * m); // Pre-cálculo geométrico

        double H = (initialGuess <= 0) ? 1.0 : initialGuess;

        for (int i = 0; i < MAX_ITERATIONS; i++) {
            // 1. Calcular Q actual con el H estimado
            double calculatedDischarge = calculateQ(H, b, m, n, sqrtSlope, pythagoras);

            // 2. Función objetivo f(H) = 0
            double f_H = calculatedDischarge - targetDischarge;

            if (Math.abs(f_H) < TOLERANCE) return (float) H;

            // 3. Derivada dQ/dH
            double dQ_dH = calculate_dQ_dH(H, b, m, calculatedDischarge, pythagoras);

            if (Math.abs(dQ_dH) < 1e-6) break; // Derivada plana

            // 4. Paso Newton
            double h_next = H - f_H / dQ_dH;

            if (Math.abs(h_next - H) < TOLERANCE) return (float) Math.max(0, h_next);

            // 5. Clamp para evitar valores negativos
            H = Math.max(0.001, h_next);
        }

        return (float) H;
    }

    // Sobrecarga para RiverFactory (sin guess inicial)
    public static float calculateNormalDepth(float targetDischarge, float slope, float n, float b, float m) {
        return calculateNormalDepth(targetDischarge, slope, n, b, m, 1.0f);
    }

    // --- Helpers Matemáticos Internos (Double precision) ---

    private static double calculateQ(double H, double b, double m, double n, double sqrtSlope, double pythagoras) {
        double A = (b + m * H) * H;
        if (A <= 0) return 0.0;
        double P = b + 2.0 * H * pythagoras;
        double R = A / P;
        return (1.0 / n) * A * Math.pow(R, 2.0 / 3.0) * sqrtSlope;
    }

    private static double calculate_dQ_dH(double H, double b, double m, double currentQ, double pythagoras) {
        double topWidth = b + 2.0 * m * H;
        double A = (b + m * H) * H;
        double P = b + 2.0 * H * pythagoras;

        // Regla de la cadena optimizada para Manning
        double term_A = (5.0 * topWidth) / (3.0 * A);
        double term_P = (4.0 * pythagoras) / (3.0 * P);

        return currentQ * (term_A - term_P);
    }

    /**
     * Calcula velocidad V = Q / A.
     */
    public static float calculateVelocity(float discharge, float depth, float b, float m) {
        if (depth <= 1e-4) return 0f;
        float area = (b + m * depth) * depth;
        return discharge / area;
    }
}