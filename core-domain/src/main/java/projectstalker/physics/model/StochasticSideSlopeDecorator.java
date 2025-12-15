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

    public StochasticSideSlopeDecorator(SpatialModel wrappedModel) {
        this.wrappedModel = wrappedModel;
    }

    @Override
    public float calculate(int cellIndex, RiverConfig config, double driverValue, double noiseFactor) {
        // 1. Obtener el talud base físico (ej: determinado por el material)
        float baseZ = wrappedModel.calculate(cellIndex, config, driverValue, noiseFactor);

        // 2. Calcular la variabilidad
        // noiseFactor viene [-1, 1].
        // sideSlopeVariability define cuánto puede desviarse el talud (ej: +/- 0.5)
        double variability = noiseFactor * config.sideSlopeVariability();

        // 3. Aplicar
        double finalZ = baseZ + variability;

        // 4. Seguridad
        // Un talud negativo es físicamente imposible (sería una cueva).
        // Un talud de 0.1 es prácticamente vertical.
        return (float) Math.max(0.1, finalZ);
    }
}