package projectstalker.physics.solver.impl;

import projectstalker.config.RiverConfig;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;
import projectstalker.physics.model.RiverTemperatureModel;
import projectstalker.physics.solver.HydrologySolver;

/**
 * Implementación del motor de hidrología que utiliza la Ecuación de Manning
 * para calcular el estado del río.
 * <p>
 * Este motor es responsable de calcular el estado completo del agua
 * (profundidad, velocidad, temperatura y pH) para el siguiente paso de tiempo.
 */

/*
    TODO Implementar celdas RESERVOIR como un único cuerpo de agua interconectado.
    La nueva cota del agua se calcula con un simple balance de volúmenes, como si fuera una bañera:

    Volumen Actual: Calcula el volumen total de agua en el embalse en el instante t (sumando el volumen de cada celda RESERVOIR).

    Caudal de Entrada (Q_in): Es el caudal que le llega de la última celda de río "normal" aguas arriba.

    Caudal de Salida (Q_out): Este es el valor crucial que será calculado por la celda DAM_STRUCTURE.

    Nuevo Volumen: El nuevo volumen en el instante t+1 será: volumen anterior + (diferencia de caudales) * periodo de evaluación

    De volumen a cota: Traducirlo de nuevo a una cota de agua con la información de GeoEvManMadeDam

    Calcular Profundidades Individuales: Finalmente, la nueva profundidad para cada celda RESERVOIR i es simplemente:
                    Profundidad = nueva cota de agua menos cota de fondo de la celda


 */

public class SequentialManningHydrologySolver implements HydrologySolver {

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
        float[] newWaterDepth = new float[cellCount];
        float[] newVelocity = new float[cellCount];
        float[] newTemperature;
        float[] newPh = new float[cellCount];


        // 1. Calcular el perfil de temperaturas usando el nuevo modelo.
        RiverTemperatureModel tempModel = new RiverTemperatureModel(config, geometry);
        newTemperature = tempModel.generateProfile(currentTimeInSeconds);

        // 2. Asignar el pH directamente desde la geometría.
        for (int i = 0; i < cellCount; i++) {
            newPh[i] = geometry.getPhAt(i);
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
                newWaterDepth[i] = 0.0F;
                newVelocity[i] = 0.0F;
                continue; // Saltar al siguiente ciclo del bucle
            }

            // 2. Encontrar la Profundidad (H) que satisface la Ecuación de Manning
            float initialGuess = currentState.getWaterDepthAt(i);
            newWaterDepth[i] = ManningEquationSolver.findDepth(
                    (float) currentDischarge,
                    initialGuess,
                    i,
                    geometry
            );

            // 3. Calcular la Velocidad (v) a partir de la nueva profundidad y el caudal
            double newArea = geometry.getCrossSectionalArea(i, newWaterDepth[i]);
            if (newArea > 1e-6) { // Evitar división por cero si el río está seco
                newVelocity[i] = (float) (currentDischarge / newArea);
            } else {
                newVelocity[i] = 0.0F;
            }
        }
        // --- Devolver el nuevo estado inmutable ---
        return RiverState.builder()
                .waterDepth(newWaterDepth)
                .velocity(newVelocity)
                .temperature(newTemperature)
                .ph(newPh)
                .contaminantConcentration(new float[cellCount])
                .build();
    }


}