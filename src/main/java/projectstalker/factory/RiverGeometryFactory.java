package projectstalker.factory;

import projectstalker.config.RiverConfig;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.utils.FastNoiseLite;

/**
 * Fábrica responsable de la creación procedural de instancias de {@link RiverGeometry}.
 * <p>
 * Encapsula la lógica compleja de generación de perfiles de río realistas,
 * asegurando que el objeto final sea siempre físicamente consistente.
 *
 * <p><b>Lógica de Generación Procedural</b></p>
 * <p>
 * La generación se basa en un enfoque de <b>doble capa de ruido procedural</b> para crear
 * variabilidad a diferentes escalas, de forma análoga a un pintor que usa un pincel
 * grueso para las formas generales y uno fino para los detalles.
 * </p>
 * <ol>
 * <li><b>Ruido Zonal (Baja Frecuencia):</b> Define las características a gran escala del
 * río. Crea "zonas" de varios kilómetros que pueden ser, por ejemplo, más tranquilas
 * y anchas (remansos) o más abruptas y estrechas (rápidos). Este ruido se utiliza para
 * <b>correlacionar propiedades físicas</b>: las zonas con pendientes altas tendrán
 * un coeficiente de Manning más alto (más rugosidad) y un cauce más estrecho.</li>
 *
 * <li><b>Ruido de Detalle (Alta Frecuencia):</b> Superpone variaciones celda a celda
 * sobre las características zonales. Esto añade la textura y la irregularidad natural
 * que se observa en un río real, evitando que los tramos largos parezcan
 * artificialmente uniformes.</li>
 * </ol>
 * <p>
 * La combinación de estas dos capas permite generar geometrías de río que no solo son
 * visualmente creíbles, sino también físicamente consistentes en sus propiedades.
 *
 * @author Duo Xu
 * @version 1.2
 * @since 2025-10-13
 */
public class RiverGeometryFactory {

    /**
     * Crea una instancia de RiverGeometry con características realistas
     * generadas proceduralmente a partir de una configuración dada.
     *
     * @param config El objeto de configuración que define las propiedades del río.
     * @return Un objeto RiverGeometry inmutable y físicamente consistente.
     */
    public RiverGeometry createRealisticRiver(RiverConfig config) {
        // 1. Inicialización
        final int cellCount = (int) Math.round(config.totalLength() / config.spatialResolution());

        // --- Generador de ruido para el detalle celda a celda ---
        final FastNoiseLite detailNoise = new FastNoiseLite((int) config.seed());
        detailNoise.SetNoiseType(FastNoiseLite.NoiseType.Perlin);
        detailNoise.SetFrequency(config.detailNoiseFrequency());

        // --- Generador de ruido para las grandes zonas (con una semilla diferente) ---
        final FastNoiseLite zoneNoise = new FastNoiseLite((int) config.seed() + 1);
        zoneNoise.SetNoiseType(FastNoiseLite.NoiseType.Perlin);
        zoneNoise.SetFrequency(config.zoneNoiseFrequency());

        // 2. Creación de los arrays de atributos
        double[] elevationProfile = new double[cellCount];
        double[] bottomWidth = new double[cellCount];
        double[] sideSlope = new double[cellCount];
        double[] manningCoefficient = new double[cellCount];
        double[] baseDecayCoefficientAt20C = new double[cellCount];
        double[] phProfile = new double[cellCount];

        // 3. Generación procedural de los perfiles celda por celda
        elevationProfile[0] = config.initialElevation();

        for (int i = 0; i < cellCount; i++) {
            double currentDetailNoise = detailNoise.GetNoise(i, 0);
            double currentZoneNoise = zoneNoise.GetNoise(i, 0);
            double zoneMultiplier = (currentZoneNoise + 1.0) / 2.0; // Mapea a [0, 1]

            // --- Generar Elevación con Perfil Cóncavo y Variabilidad Variable ---
            if (i > 0) {
                // 1. Calcular el progreso a lo largo del río (de 0.0 al inicio a 1.0 al final)
                double progress = (double) (i - 1) / (cellCount - 1);

                // 2. Definir la pendiente máxima (cabecera) y mínima (desembocadura)
                // Se usa un factor de concavidad para controlar la curvatura del perfil.
                // Si concavityFactor = 0, la pendiente es constante (perfil lineal).
                // Si concavityFactor = 0.5, la pendiente inicial es 1.5x la media y la final 0.5x la media.
                double maxSlope = config.averageSlope() * (1.0 + config.concavityFactor());
                double minSlope = config.averageSlope() * (1.0 - config.concavityFactor());

                // 3. Interpolar para obtener la pendiente base en la celda actual
                double currentBaseSlope = maxSlope - progress * (maxSlope - minSlope);
                double baseDrop = currentBaseSlope * config.spatialResolution();

                // 4. Aplicar la variabilidad local usando el ruido de zona y detalle
                double localSlopeVariability = config.slopeVariability() * zoneMultiplier;
                double noiseEffectOnDrop = currentDetailNoise * localSlopeVariability;

                // 5. Calcular la caída total, asegurando que no sea negativa
                double totalDrop = Math.max(0, baseDrop + noiseEffectOnDrop);
                elevationProfile[i] = elevationProfile[i - 1] - totalDrop;
            }

            // --- Generar Ancho del Fondo (CORRELACIONADO) ---
            // Zonas suaves (zoneMultiplier bajo) son más anchas.
            // Zonas abruptas (zoneMultiplier alto) son más estrechas.
            double widthModulation = (1.0 - zoneMultiplier) * config.widthVariability(); // Modulación inversa
            double widthValue = config.baseWidth() + widthModulation + (currentDetailNoise * config.widthVariability() * 0.2); // Añadimos un poco de detalle
            bottomWidth[i] = Math.max(0.1, widthValue);

            // --- Generar Pendiente de Taludes ---
            // La mantenemos con el ruido de detalle para simplicidad.
            double sideSlopeValue = config.baseSideSlope() + currentDetailNoise * config.sideSlopeVariability();
            sideSlope[i] = Math.max(0, sideSlopeValue);

            // --- Generar Coeficiente de Manning (CORRELACIONADO) ---
            // Zonas suaves (zoneMultiplier bajo) tienen Manning bajo.
            // Zonas abruptas (zoneMultiplier alto) tienen Manning alto.
            double manningModulation = zoneMultiplier * config.manningVariability(); // Modulación directa
            double manningValue = config.baseManning() + manningModulation + (currentDetailNoise * config.manningVariability() * 0.2);
            manningCoefficient[i] = Math.max(0.01, manningValue);

            // --- Generar Coeficiente de Reacción y pH ---
            // Estos son menos dependientes de la pendiente, así que los mantenemos con el ruido de detalle.
            double decayValue = config.baseDecayRateAt20C() + currentDetailNoise * config.decayRateVariability();
            baseDecayCoefficientAt20C[i] = Math.max(0.0, decayValue);

            double phValue = config.basePh() + currentDetailNoise * config.phVariability();
            phProfile[i] = Math.max(6.0, Math.min(9.0, phValue));
        }

        // 4. Instanciar y devolver el objeto RiverGeometry final y validado
        return new RiverGeometry(
                cellCount,
                config.spatialResolution(),
                elevationProfile,
                bottomWidth,
                sideSlope,
                manningCoefficient,
                baseDecayCoefficientAt20C,
                phProfile
        );
    }
}