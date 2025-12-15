package projectstalker.physics.model;

import projectstalker.config.RiverConfig;

/**
 * Modelo Base de Taludes (Side Slope) basado en Material (Ángulo de Reposo).
 * <p>
 * Establece la inclinación de las orillas z basándose en la rugosidad del lecho (Manning),
 * asumiendo que el Manning es un indicador del tipo de material geológico.
 * <ul>
 * <li><b>Manning Alto (Roca/Grava gruesa):</b> Material cohesivo o duro. Soporta taludes verticales o empinados ($z$ bajo).</li>
 * <li><b>Manning Bajo (Arena/Limo):</b> Material suelto. Requiere taludes suaves para ser estable ($z$ alto).</li>
 * </ul>
 */
public class MaterialBasedSideSlopeModel implements SpatialModel {

    // Manning de referencia para interpolación
    private static final double MANNING_ROCK = 0.050; // Roca
    private static final double MANNING_SAND = 0.025; // Arena fina

    // Taludes de referencia (Horizontal : Vertical)
    private static final double Z_ROCK = 0.5; // Casi vertical
    private static final double Z_SAND = 4.0; // Muy plano

    @Override
    public float calculate(int cellIndex, RiverConfig config, double localManning, double unusedNoise) {
        // Interpolación Lineal Inversa
        // A mayor Manning (más duro), menor Z (más vertical).

        // 1. Normalizar Manning dentro del rango esperado
        // Clampeamos para no extrapolar valores locos fuera de arena-roca
        double n = Math.max(MANNING_SAND, Math.min(MANNING_ROCK, localManning));

        // 2. Calcular factor de posición (0.0 = Arena, 1.0 = Roca)
        double ratio = (n - MANNING_SAND) / (MANNING_ROCK - MANNING_SAND);

        // 3. Interpolar Z
        // Si ratio es 0 (Arena) -> Z_SAND
        // Si ratio es 1 (Roca)  -> Z_ROCK
        double z = Z_SAND - (ratio * (Z_SAND - Z_ROCK));

        return (float) z;
    }
}