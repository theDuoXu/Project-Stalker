package projectstalker.physics.impl;

import projectstalker.config.RiverConfig;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;
import projectstalker.physics.solver.IHydrologySolver;

/**
 * Implementación del motor de hidrología que utiliza la Ecuación de Manning
 * para calcular el estado del río.
 * <p>
 * Este motor es responsable de calcular el estado completo del agua
 * (profundidad, velocidad, temperatura y pH) para el siguiente paso de tiempo.
 */
public class ManningHydrologySolver implements IHydrologySolver {

    private static final double SECONDS_IN_A_DAY = 24.0 * 3600.0;
    private static final double DAYS_IN_A_YEAR = 365.25;

    /**
     * Calcula el siguiente estado del río a partir del estado actual.
     *
     * @param currentState         El estado del río en el tiempo 't'.
     * @param geometry             La geometría inmutable del cauce del río.
     * @param config               La configuración global de la simulación.
     * @param currentTimeInSeconds El tiempo absoluto de la simulación en segundos.
     * @param inputDischarge       El caudal de entrada en el inicio del río en m³/s.
     * @return Un nuevo objeto {@link RiverState} que representa el estado en el tiempo 't+1'.
     */
    @Override
    public RiverState calculateNextState(
            RiverState currentState,
            RiverGeometry geometry,
            RiverConfig config,
            double currentTimeInSeconds,
            double inputDischarge) {

        final int cellCount = geometry.getCellCount();

        // --- Validación de Consistencia del Sistema ---
        if (currentState.waterDepth().length != cellCount) {
            throw new IllegalStateException(
                    "Inconsistencia de estado: El RiverState tiene " + currentState.waterDepth().length +
                            " celdas, pero el RiverGeometry tiene " + cellCount + " celdas."
            );
        }

        // --- Arrays para el nuevo estado ---
        double[] newWaterDepth = new double[cellCount];
        double[] newVelocity = new double[cellCount];
        double[] newTemperature = new double[cellCount];
        double[] newPh = new double[cellCount];


        // --- PASO A: Calcular Temperatura y pH ---
        // Estos valores dependen del tiempo, pero son uniformes a lo largo del río en este modelo.
        final double dayOfYear = (currentTimeInSeconds / SECONDS_IN_A_DAY) % DAYS_IN_A_YEAR;
        final double secondOfDay = currentTimeInSeconds % SECONDS_IN_A_DAY;

        // 1. Ciclo Estacional (sinusoide con período de 1 año)
        final double seasonalCycle = Math.sin((dayOfYear / DAYS_IN_A_YEAR) * 2.0 * Math.PI);
        final double baseSeasonalTemp = config.averageAnnualTemperature() + config.seasonalTempVariation() * seasonalCycle;

        // 2. Ciclo Diario (sinusoide con período de 24 horas)
        final double dailyCycle = Math.sin((secondOfDay / SECONDS_IN_A_DAY) * 2.0 * Math.PI);
        final double finalTemp = baseSeasonalTemp + config.dailyTempVariation() * dailyCycle;

        // Asignamos los valores de temperatura y pH a todas las celdas
        for (int i = 0; i < cellCount; i++) {
            newTemperature[i] = finalTemp;
            newPh[i] = geometry.getPhAt(i); // Leyendo el ph base desde la geometría del río
        }


        // --- PASO B: Calcular Hidráulica (Profundidad y Velocidad) ---
        for (int i = 0; i < cellCount; i++) {
            // 1. Determinar el Caudal (Q) que entra en la celda 'i'
            final double currentDischarge;
            if (i == 0) {
                currentDischarge = inputDischarge;
            } else {
                // El caudal que entra es el que salió de la celda anterior en el estado previo
                double previousArea = geometry.getCrossSectionalArea(i - 1, currentState.getWaterDepthAt(i - 1));
                currentDischarge = previousArea * currentState.getVelocityAt(i - 1);
            }

            // Si el caudal de entrada es prácticamente cero, el río está seco en este punto.
            if (currentDischarge < 1e-6) {
                newWaterDepth[i] = 0.0;
                newVelocity[i] = 0.0;
                continue; // Saltar al siguiente ciclo del bucle
            }

            // 2. Encontrar la Profundidad (H) que satisface la Ecuación de Manning
            double initialGuess = currentState.getWaterDepthAt(i);
            newWaterDepth[i] = ManningEquationSolver.findDepth(
                    currentDischarge,
                    initialGuess,
                    i,
                    geometry
            );

            // 3. Calcular la Velocidad (v) a partir de la nueva profundidad y el caudal
            double newArea = geometry.getCrossSectionalArea(i, newWaterDepth[i]);
            if (newArea > 1e-6) { // Evitar división por cero si el río está seco
                newVelocity[i] = currentDischarge / newArea;
            } else {
                newVelocity[i] = 0.0;
            }
        }
        // --- Devolver el nuevo estado inmutable ---
        return new RiverState(newWaterDepth, newVelocity, newTemperature, newPh);
    }

    /**
     * Clase anidada estática que encapsula la lógica para resolver la ecuación de Manning.
     * Utiliza el method de Newton-Raphson para encontrar la profundidad (H)
     * a partir de un caudal (Q) dado.
     */
    private static final class ManningEquationSolver {

        private static final int MAX_ITERATIONS = 20;
        private static final double TOLERANCE = 1e-6; // Tolerancia para convergencia en metros

        /**
         * Encuentra la profundidad del agua (H) que corresponde a un caudal dado.
         * Esta clase es Thread safe
         *
         * @param targetDischarge El caudal objetivo que la celda debe evacuar (m³/s).
         * @param initialDepthGuess Una estimación inicial para la profundidad (m).
         * @param cellIndex El índice de la celda del río.
         * @param geometry La geometría del río.
         * @return La profundidad calculada (m).
         */
        public static double findDepth(double targetDischarge, double initialDepthGuess, int cellIndex, RiverGeometry geometry) {

            // Extraer parámetros geométricos una sola vez
            final double b = geometry.cloneBottomWidth()[cellIndex];
            final double m = geometry.cloneSideSlope()[cellIndex];
            final double n = geometry.getManningAt(cellIndex);
            double S = geometry.getBedSlopeAt(cellIndex);

            // Evitar pendiente cero o negativa que impide el cálculo.
            if (S <= 1e-7) {
                S = 1e-7;
            }

            double H = (initialDepthGuess <= 0) ? 0.1 : initialDepthGuess; // Empezar con un valor pequeño si la estimación es cero

            for (int i = 0; i < MAX_ITERATIONS; i++) {
                // Función f(H) = Q_calculado(H) - Q_objetivo
                double calculatedDischarge = calculateQ(H, b, m, n, S);
                double f_H = calculatedDischarge - targetDischarge;

                // Si el error es suficientemente pequeño, hemos convergido.
                if (Math.abs(f_H) < TOLERANCE) {
                    return H;
                }

                // Derivada f'(H) = dQ/dH
                double dQ_dH = calculate_dQ_dH(H, b, m, n, S, calculatedDischarge);

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
        private static double calculateQ(double H, double b, double m, double n, double S) {
            double A = (b + m * H) * H;
            if (A <= 0) return 0.0;

            double P = b + 2.0 * H * Math.sqrt(1.0 + m * m);
            if (P <= 1e-9) return 0.0;

            double R = A / P;
            return (1.0 / n) * A * Math.pow(R, 2.0 / 3.0) * Math.sqrt(S);
        }

        /**
         * Calcula la derivada del caudal respecto a la profundidad (dQ/dH) analíticamente.
         */
        private static double calculate_dQ_dH(double H, double b, double m, double n, double S, double currentQ) {
            if (H <= 1e-9) {
                return Double.POSITIVE_INFINITY;
            }

            // Términos de la derivada del área y perímetro (Ver en los apuntes)
            double term_A_derivative = (5.0 * (b + 2.0 * m * H)) / ((b + m * H) * H);
            double term_P_derivative = (4.0 * Math.sqrt(1.0 + m * m)) / (b + 2.0 * H * Math.sqrt(1.0 + m * m));

            return (currentQ / 3.0) * (term_A_derivative - term_P_derivative);
        }
    }
}