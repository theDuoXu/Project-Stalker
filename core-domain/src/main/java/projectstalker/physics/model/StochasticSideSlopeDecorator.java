package projectstalker.physics.model;

import projectstalker.config.RiverConfig;

/**
 * Decorador Estocástico para Taludes.
 * <p>
 * Añade variabilidad natural a la inclinación de las orillas calculada por el modelo base.
 * Simula efectos locales como:
 * <ul>
 * <li>Erosión diferencial en meandros.</li>
 * <li>Vegetación de ribera (raíces) que sostienen taludes más verticales.</li>
 * <li>Derrumbes locales.</li>
 * </ul>
 */
public class StochasticSideSlopeDecorator implements SpatialModel {

    private final SpatialModel wrappedModel;

    // Define el Ruido Máximo permitido como porcentaje del valor base.
    // 0.25 significa que el valor puede variar un +/- 25%.
    // Esto preserva la tendencia: un talud vertical (0.5) variará poco (±0.12),
    // y uno suave (4.0) variará más (±1.0), manteniendo la física coherente.
    private static final double NOISE_PERCENTAGE = 0.25;

    public StochasticSideSlopeDecorator(SpatialModel wrappedModel) {
        this.wrappedModel = wrappedModel;
    }

    @Override
    public float calculate(int cellIndex, RiverConfig config, double driverValue, double noiseFactor) {
        // 1. Obtener el talud base físico (Determinista)
        float baseZ = wrappedModel.calculate(cellIndex, config, driverValue, noiseFactor);

        // 2. Calcular la perturbación relativa
        // noiseFactor rango: [-1.0, 1.0]
        double perturbation = baseZ * noiseFactor * NOISE_PERCENTAGE;

        // 3. Aplicar
        double finalZ = baseZ + perturbation;

        // 4. Seguridad (Clamping)
        // Evitamos valores negativos o cero absoluto.
        return (float) Math.max(0.1, finalZ);
    }
}