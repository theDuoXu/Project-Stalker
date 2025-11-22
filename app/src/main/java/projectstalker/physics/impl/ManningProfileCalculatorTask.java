// Fichero: projectstalker/physics/impl/ManningProfileCalculatorTask.java
package projectstalker.physics.impl;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.With;
import lombok.extern.slf4j.Slf4j;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.physics.solver.ManningEquationSolver;

import java.util.concurrent.Callable;

/**
 * Tarea ejecutable (Runnable) que calcula un perfil hidrológico completo
 * (profundidades y velocidades) para un único conjunto de caudales de entrada.
 * Está diseñado para ser ejecutado en un pool de hilos.
 */
@Slf4j
@Getter
@RequiredArgsConstructor
public class ManningProfileCalculatorTask implements Callable<ManningProfileCalculatorTask> {

    // --- Entradas para la tarea ---
    private final float[] targetDischarges; // Caudales para cada celda de ESTA simulación
    private final float[] initialDepthGuess; // Profundidad inicial para cada celda
    private final RiverGeometry geometry; // Geometría del río (compartida)

    // --- Resultados de la tarea ---
    private float[] calculatedWaterDepth;
    private float[] calculatedVelocity;

    @Override
    public ManningProfileCalculatorTask call() throws Exception {
        int cellCount = geometry.getCellCount();
        this.calculatedWaterDepth = new float[cellCount];
        this.calculatedVelocity = new float[cellCount];

        for (int i = 0; i < cellCount; i++) {
            final float currentDischarge = targetDischarges[i];

            // Si el caudal es insignificante, la celda está seca.
            if (currentDischarge < 1e-6) {
                calculatedWaterDepth[i] = 0.0F;
                calculatedVelocity[i] = 0.0F;
                continue;
            }

            // Encontrar la profundidad que satisface la ecuación de Manning
            calculatedWaterDepth[i] = ManningEquationSolver.findDepth(
                    currentDischarge,
                    initialDepthGuess[i], // Usar la estimación para esta celda
                    i,
                    geometry
            );

            // Calcular la velocidad a partir de la nueva profundidad
            double newArea = geometry.getCrossSectionalArea(i, calculatedWaterDepth[i]);
            if (newArea > 1e-6) {
                calculatedVelocity[i] = (float) (currentDischarge / newArea);
            } else {
                calculatedVelocity[i] = 0.0F;
            }
        }
        return this;
    }
}