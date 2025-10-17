// Fichero: projectstalker/physics/impl/ManningProfileCalculatorTask.java
package projectstalker.physics.impl;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.With;
import lombok.extern.slf4j.Slf4j;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.physics.solver.ManningEquationSolver;

/**
 * Tarea ejecutable (Runnable) que calcula un perfil hidrológico completo
 * (profundidades y velocidades) para un único conjunto de caudales de entrada.
 * Está diseñado para ser ejecutado en un pool de hilos.
 */
@Slf4j
@Getter
@With
@RequiredArgsConstructor
public class ManningProfileCalculatorTask implements Runnable {

    // --- Entradas para la tarea ---
    private final double[] targetDischarges; // Caudales para cada celda de ESTA simulación
    private final double[] initialDepthGuess; // Profundidad inicial para cada celda
    private final RiverGeometry geometry; // Geometría del río (compartida)

    // --- Resultados de la tarea ---
    private final double[] calculatedWaterDepth;
    private final double[] calculatedVelocity;

    public ManningProfileCalculatorTask(double[] targetDischarges, double[] initialDepthGuess, RiverGeometry geometry) {
        this.targetDischarges = targetDischarges;
        this.initialDepthGuess = initialDepthGuess;
        this.geometry = geometry;

        int cellCount = geometry.getCellCount();
        this.calculatedWaterDepth = new double[cellCount];
        this.calculatedVelocity = new double[cellCount];
    }

    @Override
    public void run() {
        int cellCount = geometry.getCellCount();

        for (int i = 0; i < cellCount; i++) {
            final double currentDischarge = targetDischarges[i];

            // Si el caudal es insignificante, la celda está seca.
            if (currentDischarge < 1e-6) {
                calculatedWaterDepth[i] = 0.0;
                calculatedVelocity[i] = 0.0;
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
                calculatedVelocity[i] = currentDischarge / newArea;
            } else {
                calculatedVelocity[i] = 0.0;
            }
        }
    }
}