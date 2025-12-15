package projectstalker.physics.model;

import projectstalker.config.RiverConfig;

/**
 * Modelo de pH basado en Zonas Geológicas.
 * <p>
 * El pH del agua está fuertemente determinado por la litología de la cuenca.
 * Este modelo usa el ruido de baja frecuencia (Macro-escala) para simular
 * transiciones entre diferentes tipos de roca madre.
 * <ul>
 * <li><b>Zona Positiva (> 0):</b> Terrenos básicos (ej: Caliza). pH tiende a subir.</li>
 * <li><b>Zona Negativa (< 0):</b> Terrenos ácidos (ej: Granito/Turba). pH tiende a bajar.</li>
 * </ul>
 */
public class GeologyBasedPhModel implements SpatialModel {

    /**
     * Define cuánto peso tiene la geología (zona) frente a la variabilidad local.
     * 0.8 significa que el 80% de la variación del pH se debe al tipo de roca,
     * y solo el 20% a factores locales (ruido detalle).
     */
    private static final double GEOLOGY_WEIGHT = 0.8;
    private static final double LOCAL_WEIGHT = 0.2;

    /**
     * @param cellIndex   Índice de la celda.
     * @param config      Configuración global.
     * @param geologyZone Valor del "Ruido Zonal" (driverValue). [-1.0 a 1.0].
     * @param localNoise  Valor del "Ruido Detalle" (noiseFactor). [-1.0 a 1.0].
     * @return El valor de pH calculado.
     */
    @Override
    public float calculate(int cellIndex, RiverConfig config, double geologyZone, double localNoise) {
        float basePh = config.basePh();
        float maxVariability = config.phVariability(); // 1.5 (significa +/- 1.5 pH)

        // 1. Calcular el efecto Geológico (Macro escala)
        // Si geologyZone es 1.0 (Caliza pura), subimos el pH.
        // Si geologyZone es -1.0 (Turba), bajamos el pH.
        double geologicalShift = geologyZone * maxVariability * GEOLOGY_WEIGHT;

        // 2. Calcular el efecto Local (Micro escala)
        // Pequeñas variaciones por vertidos, vegetación en descomposición local, etc.
        double localShift = localNoise * maxVariability * LOCAL_WEIGHT;

        // 3. Combinar
        double finalPh = basePh + geologicalShift + localShift;

        // 4. Clamping de seguridad (Rango natural del agua de río: 4.0 - 10.0)
        // Valores fuera de esto son extremadamente raros en la naturaleza sin contaminación industrial.
        return (float) Math.max(4.0, Math.min(10.0, finalPh));
    }
}