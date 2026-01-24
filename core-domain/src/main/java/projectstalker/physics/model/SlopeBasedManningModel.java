package projectstalker.physics.model;

import projectstalker.config.RiverConfig;
/**
 * Modelo de Rugosidad basado en Competencia de Flujo (Sediment Sorting).
 * <p>
 * Modela cómo la pendiente local determina el tipo de material del lecho:
 * <ul>
 * <li><b>Pendiente > Media:</b> Alta energía cinética. Erosión de finos.
 * Lecho de rocas/cantos rodados -> <b>Mayor Manning</b>.</li>
 * <li><b>Pendiente menor que Media:</b> Baja energía cinética. Sedimentación.
 * Lecho de arena/limo -> <b>Menor Manning</b>.</li>
 * </ul>
 * <p>
 * Este modelo elimina la necesidad de simular artificialmente la suavización
 * del río hacia la desembocadura, ya que emerge naturalmente de la reducción de la pendiente.
 */
public class SlopeBasedManningModel implements SpatialModel {

    /**
     * Límites de seguridad para evitar coeficientes físicos imposibles.
     * 0.015 = Hormigón/Barro muy liso.
     * 0.150 = Caos absoluto (troncos, rocas gigantes).
     */
    private static final float MIN_SAFE_MANNING = 0.015f;
    private static final float MAX_SAFE_MANNING = 0.150f;

    @Override
    public float calculate(int cellIndex, RiverConfig config, double localSlope, double noiseFactor) {
        double baseManning = config.baseManning();
        double avgSlope = config.averageSlope();

        // 1. Calcular Ratio de Energía (Pendiente Local / Media)
        double safeLocalSlope = Math.max(EPSILON, localSlope);
        double slopeRatio = safeLocalSlope / avgSlope;

        // 2. Aplicar Ley de Potencia (Relación Directa)
        // Si slopeRatio > 1 (Más inclinado) -> Factor > 1 (Más rugoso)
        // Si slopeRatio < 1 (Más plano)     -> Factor < 1 (Más suave)
        double sedimentFactor = Math.pow(slopeRatio, config.roughnessSensitivity());

        // Acotamos el factor para que la geología no multiplique el Manning x10
        // Rango dinámico permitido: 0.5x (mitad de rugoso) a 2.5x (más del doble)
        sedimentFactor = Math.max(0.5, Math.min(2.5, sedimentFactor));

        // 3. Aplicar Variabilidad Estocástica (Ruido)
        // Aquí el ruido representa vegetación local, obstáculos caídos, etc.
        // noiseFactor viene normalizado (aprox -1 a 1).
        double localVariability = noiseFactor * config.manningVariability();

        // 4. Cálculo final
        double finalManning = (baseManning * sedimentFactor) + localVariability;

        // 5. Clamping de seguridad
        return (float) Math.max(MIN_SAFE_MANNING, Math.min(MAX_SAFE_MANNING, finalManning));
    }
}