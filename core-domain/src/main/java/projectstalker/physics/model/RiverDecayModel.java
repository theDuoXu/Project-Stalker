package projectstalker.physics.model;

import projectstalker.domain.river.RiverGeometry;

/**
 * Modelo de Evolución Temporal del Coeficiente de Reacción (k).
 * <p>
 * Aplica la corrección térmica de Arrhenius sobre el perfil espacial base.
 * Este modelo es dependiente del modelo de temperatura.
 */
public class RiverDecayModel implements TimeEvolutionModel {

    // Constante térmica (Theta) estándar para degradación de DBO (Materia Orgánica).
    // Valores típicos: 1.047 para DBO carbonácea.
    private static final double THETA_ARRHENIUS = 1.047;

    private final RiverGeometry geometry;
    private final TemperatureModel temperatureModel;

    public RiverDecayModel(RiverGeometry geometry, TemperatureModel temperatureModel) {
        this.geometry = geometry;
        this.temperatureModel = temperatureModel;
    }

    @Override
    public float[] generateProfile(double currentTimeInSeconds) {
        // Delegamos en el modelo de temperatura inyectado.
        float[] currentTemperatures = temperatureModel.generateProfile(currentTimeInSeconds);
        return generateProfile(currentTemperatures);
    }

    /**
     * Versión con temperaturas precalculadas
     */
    public float[] generateProfile(float[] currentTemperatures) {
        final int cellCount = geometry.getCellCount();
        final float[] currentDecayProfile = new float[cellCount];

        // 1. Obtener la base espacial (calculada por ManningBasedDecayModel en la Factory)
        // Estos valores son k a 20°C.
        float[] baseDecayAt20 = geometry.getBaseDecayCoefficientAt20C();

        // 2. Aplicar corrección de Arrhenius celda a celda
        for (int i = 0; i < cellCount; i++) {
            float k20 = baseDecayAt20[i];
            float temp = currentTemperatures[i];

            // Fórmula: k(T) = k(20) * Theta^(T - 20)
            // Si T > 20, k aumenta. Si T < 20, k disminuye.
            double thermalFactor = Math.pow(THETA_ARRHENIUS, temp - 20.0);

            currentDecayProfile[i] = (float) (k20 * thermalFactor);
        }

        return currentDecayProfile;
    }
}