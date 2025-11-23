package projectstalker.factory;

import projectstalker.config.RiverConfig;
import projectstalker.domain.event.GeologicalEvent;
import projectstalker.domain.river.RiverSectionType;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.utils.FastNoiseLite;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

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
     * Define la separación mínima requerida entre los puntos centrales
     * de dos eventos geológicos consecutivos.
     */
    private static final int MINIMUM_CELL_SEPARATION = 5;

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
        float[] elevationProfile = new float[cellCount];
        float[] bottomWidth = new float[cellCount];
        float[] sideSlope = new float[cellCount];
        float[] manningCoefficient = new float[cellCount];
        float[] baseDecayCoefficientAt20C = new float[cellCount];
        float[] phProfile = new float[cellCount];
        float[] dispersionAlpha = new float[cellCount];
        RiverSectionType[] sectionTypes = new RiverSectionType[cellCount];
        Arrays.fill(sectionTypes, RiverSectionType.NATURAL);

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
                elevationProfile[i] = (float) (elevationProfile[i - 1] - totalDrop);
            }

            // --- Generar Ancho del Fondo (CORRELACIONADO) ---
            // Zonas suaves (zoneMultiplier bajo) son más anchas.
            // Zonas abruptas (zoneMultiplier alto) son más estrechas.
            double widthModulation = (1.0 - zoneMultiplier) * config.widthVariability(); // Modulación inversa
            double widthValue = config.baseWidth() + widthModulation + (currentDetailNoise * config.widthVariability() * 0.2); // Añadimos un poco de detalle
            bottomWidth[i] = (float) Math.max(0.1, widthValue);

            // --- Generar Pendiente de Taludes ---
            // La mantenemos con el ruido de detalle para simplicidad.
            double sideSlopeValue = config.baseSideSlope() + currentDetailNoise * config.sideSlopeVariability();
            sideSlope[i] = (float) Math.max(0, sideSlopeValue);

            // --- Generar Coeficiente de Manning (CORRELACIONADO) ---
            // Zonas suaves (zoneMultiplier bajo) tienen Manning bajo.
            // Zonas abruptas (zoneMultiplier alto) tienen Manning alto.
            double manningModulation = zoneMultiplier * config.manningVariability(); // Modulación directa
            double manningValue = config.baseManning() + manningModulation + (currentDetailNoise * config.manningVariability() * 0.2);
            manningCoefficient[i] = (float) Math.max(0.01, manningValue);

            // --- Generar Coeficiente de Reacción y pH ---
            // Estos son menos dependientes de la pendiente, así que los mantenemos con el ruido de detalle.
            double decayValue = config.baseDecayRateAt20C() + currentDetailNoise * config.decayRateVariability();
            baseDecayCoefficientAt20C[i] = (float) Math.max(0.0, decayValue);

            double phValue = config.basePh() + currentDetailNoise * config.phVariability();
            phProfile[i] = (float) Math.max(6.0, Math.min(9.0, phValue));

            // --- Generar Coeficiente de Dispersión (Alpha) ---
            // La dispersión aumenta en zonas rugosas (Manning alto) y con el "caos" del río.
            // Usaremos el ruido zonal: zonas 'activas' dispersan más.

            // Entre 5.0 y 20.0 para ríos naturales típicos usando la fórmula de Taylor.
            double baseAlpha = config.baseDispersionAlpha();
            double alphaVariability = config.alphaVariability();

            // Correlación: Si el Manning es alto (zona rugosa), alpha tiende a subir.
            // Usamos manningModulation (que ya calculaste arriba) como factor de influencia.
            double alphaValue = baseAlpha + (manningModulation * 100.0) + (currentDetailNoise * alphaVariability);

            // Alpha nunca debe ser negativo (físicamente imposible) ni cero (siempre hay algo de mezcla).
            dispersionAlpha[i] = (float) Math.max(0.1, alphaValue);
        }

        // 4. Instanciar y devolver el objeto RiverGeometry final y validado
        return RiverGeometry.builder()
                .cellCount(cellCount)
                .spatialResolution(config.spatialResolution())
                .elevationProfile(elevationProfile)
                .bottomWidth(bottomWidth)
                .sideSlope(sideSlope)
                .manningCoefficient(manningCoefficient)
                .baseDecayCoefficientAt20C(baseDecayCoefficientAt20C)
                .phProfile(phProfile)
                .sectionTypes(sectionTypes)
                .dispersionAlpha(dispersionAlpha)
                .build()
                ;
    }

    /**
     * Aplica una lista de eventos geológicos en serie a una geometría de río base.
     * <p>
     * El method es inmutable: no modifica el río original, sino que devuelve una
     * nueva instancia con los cambios acumulados. Primero, valida que los eventos
     * estén suficientemente separados para evitar interacciones complejas.
     *
     * @param baseRiver El río natural sobre el cual se aplicarán los eventos.
     * @param events    La lista de eventos geológicos a aplicar.
     * @return Un nuevo objeto RiverGeometry que refleja las modificaciones.
     * @throws IllegalArgumentException si dos eventos están más cerca que
     *                                  la separación mínima definida por {@code MINIMUM_CELL_SEPARATION}.
     */
    public RiverGeometry applyGeologicalEvents(RiverGeometry baseRiver, List<GeologicalEvent> events) {
        // --- 1. Manejar casos triviales ---
        if (events == null || events.isEmpty()) {
            return baseRiver; // No hay nada que hacer, devolver el río original.
        }

        // --- 2. Validar la separación entre eventos ---
        validateEventSeparation(baseRiver.getSpatialResolution(), events);

        // --- 3. Preparar datos mutables (clonación para inmutabilidad) ---
        // Obtenemos los arrays del río base. La clase RiverGeometry ya los devuelve clonados,
        // pero para ser explícitos y seguros, los clonamos de nuevo aquí.
        float[] newElevationProfile = baseRiver.cloneElevationProfile();
        float[] newBottomWidth = baseRiver.cloneBottomWidth();
        float[] newManningCoefficient = baseRiver.cloneManningCoefficient();
        RiverSectionType[] newSectionTypes = baseRiver.cloneSectionTypes();

        // --- 4. Aplicar cada evento en serie sobre los datos clonados ---
        // El polimorfismo se encarga de ejecutar la lógica correcta para cada tipo de evento.
        for (GeologicalEvent event : events) {
            event.apply(
                    baseRiver.getSpatialResolution(),
                    newElevationProfile,
                    newBottomWidth,
                    newManningCoefficient,
                    newSectionTypes
            );
        }

        // --- 5. Construir y devolver el nuevo objeto RiverGeometry final ---
        // El constructor validará la consistencia física del resultado final.
        return baseRiver.withSectionTypes(newSectionTypes);
    }

    /**
     * Method de utilidad para verificar que los eventos en una lista mantienen una
     * distancia mínima entre ellos.
     */
    private void validateEventSeparation(double spatialResolution, List<GeologicalEvent> events) {
        if (events.size() <= 1) {
            return; // No hay nada que comparar.
        }

        // Ordenar los eventos por su posición para facilitar la comparación sin modificar el original
        List<GeologicalEvent> sortedEvents = new ArrayList<>(events);
        sortedEvents.sort(Comparator.comparingDouble(GeologicalEvent::getPosition));

        // Convertir la primera posición a índice de celda.
        int lastEventCell = (int) Math.round(events.get(0).getPosition() / spatialResolution);

        // Comparar cada evento con el anterior.
        for (int i = 1; i < events.size(); i++) {
            int currentEventCell = (int) Math.round(events.get(i).getPosition() / spatialResolution);
            if (Math.abs(currentEventCell - lastEventCell) < MINIMUM_CELL_SEPARATION) {
                throw new IllegalArgumentException(
                        String.format(
                                "Los eventos geológicos están demasiado juntos. El evento en la posición %.2fm (celda %d) " +
                                        "está a menos de %d celdas del evento en %.2fm (celda %d).",
                                events.get(i).getPosition(), currentEventCell,
                                MINIMUM_CELL_SEPARATION,
                                events.get(i - 1).getPosition(), lastEventCell
                        )
                );
            }
            lastEventCell = currentEventCell;
        }
    }
}