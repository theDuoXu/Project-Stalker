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
                double previousArea = geometry.getCrossSectionalArea(i - 1, currentState.waterDepth()[i - 1]);
                currentDischarge = previousArea * currentState.velocity()[i - 1];
            }

            // 2. Encontrar la Profundidad (H) que satisface la Ecuación de Manning para el caudal actual
            // ESTA ES LA PARTE COMPLEJA QUE IMPLEMENTAREMOS MÁS ADELANTE
            // TODO: Implementar un solver numérico (ej. Newton-Raphson) para encontrar 'H'
            //       tal que Q_calculado(H) == currentDischarge.
            // Por ahora, usamos un placeholder simple: asumimos que la profundidad no cambia.
            newWaterDepth[i] = currentState.waterDepth()[i];

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
}