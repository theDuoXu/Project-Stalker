package projectstalker.factory;

import projectstalker.config.RiverConfig;
import projectstalker.domain.event.GeologicalEvent;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverSectionType;
import projectstalker.physics.model.*;
import projectstalker.utils.FastNoiseLite;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**
 * Fábrica responsable de la creación procedural de instancias de {@link RiverGeometry}.
 * <p>
 * Actúa como orquestador de las reglas físicas definidas en {@link projectstalker.physics.model}.
 * Implementa un proceso de generación en cascada donde:
 * <ol>
 * <li>Se genera la Topografía (Elevación/Pendiente) usando ruido procedural.</li>
 * <li>La Geometría Hidráulica (Ancho) reacciona a la pendiente.</li>
 * <li>La Rugosidad (Manning) reacciona a la pendiente (clasificación de sedimentos).</li>
 * <li>La Bioquímica (Decay) reacciona a la rugosidad (turbulencia).</li>
 * <li>El pH reacciona a la geología zonal.</li>
 * </ol>
 *
 * @author Duo Xu
 * @version 2.0 (Refactorizado con Patrón Estrategia)
 * @since 2025-10-13
 */
public class RiverGeometryFactory {

    private static final int MINIMUM_CELL_SEPARATION = 5;

    /**
     * Crea una instancia de RiverGeometry con características realistas
     * generadas mediante modelos físicos correlacionados.
     *
     * @param config El objeto de configuración que define las propiedades del río.
     * @return Un objeto RiverGeometry inmutable y físicamente consistente.
     */
    public static RiverGeometry createRealisticRiver(RiverConfig config) {
        // 1. Inicialización de Constantes y Generadores de Ruido
        final int cellCount = Math.round(config.totalLength() / config.spatialResolution());
        final float dx = config.spatialResolution();

        // --- A. Ruido de Detalle (Micro Frecuencia: textura, variabilidad local) ---
        final FastNoiseLite detailNoise = new FastNoiseLite((int) config.seed());
        detailNoise.SetNoiseType(FastNoiseLite.NoiseType.Perlin);
        detailNoise.SetFrequency(config.detailNoiseFrequency());

        // --- B. Ruido Principal (Meso Frecuencia: meandros, estructura del canal) ---
        final FastNoiseLite mainNoise = new FastNoiseLite((int) config.seed() + 1);
        mainNoise.SetNoiseType(FastNoiseLite.NoiseType.Perlin);
        mainNoise.SetFrequency(config.noiseFrequency());

        // --- C. Ruido Zonal (Macro Frecuencia: tipos de roca, grandes valles) ---
        final FastNoiseLite zoneNoise = new FastNoiseLite((int) config.seed() + 2);
        zoneNoise.SetNoiseType(FastNoiseLite.NoiseType.Perlin);
        zoneNoise.SetFrequency(config.zoneNoiseFrequency());

        // 2. Instanciación de TODAS las Estrategias Físicas
        // -------------------------------------------------
        SpatialModel widthModel = new DownstreamWideningDecorator(new SlopeBasedWidthModel());
        SpatialModel manningModel = new SlopeBasedManningModel();
        SpatialModel decayModel = new ManningBasedDecayModel();
        SpatialModel phModel = new GeologyBasedPhModel();

        SpatialModel sideSlopeModel = new MaterialBasedSideSlopeModel();
        sideSlopeModel = new StochasticSideSlopeDecorator(sideSlopeModel);

        SpatialModel dispersionModel = new ManningBasedDispersionModel();

        // 3. Preparación de Arrays
        float[] elevationProfile = new float[cellCount];
        float[] bottomWidth = new float[cellCount];
        float[] sideSlope = new float[cellCount];
        float[] manningCoefficient = new float[cellCount];
        float[] baseDecayCoefficientAt20C = new float[cellCount];
        float[] phProfile = new float[cellCount];
        float[] dispersionAlpha = new float[cellCount];
        RiverSectionType[] sectionTypes = new RiverSectionType[cellCount];
        Arrays.fill(sectionTypes, RiverSectionType.NATURAL);

        // Condición inicial
        elevationProfile[0] = config.initialElevation();

        // 4. Bucle Principal de Generación (Cálculo en Cascada)
        for (int i = 0; i < cellCount; i++) {
            // --- A. Obtener Ruidos ---
            double nDetail = detailNoise.GetNoise(i, 0);
            double nMain = mainNoise.GetNoise(i, 0);
            double nZone = zoneNoise.GetNoise(i, 0);

            // Factor Macro para la elevación (Combina Zona y Main)
            double combinedMacroNoise = (nZone * 0.6) + (nMain * 0.4);
            double macroMultiplier = (combinedMacroNoise + 1.0) / 2.0;

            // --- B. Cálculo del Driver Principal: Pendiente Local ---
            double currentLocalSlope;

            if (i > 0) {
                // Cálculo de Elevación (Perfil Cóncavo + Variabilidad)
                double progress = (double) (i - 1) / (cellCount - 1);
                double maxSlope = config.averageSlope() * (1.0 + config.concavityFactor());
                double minSlope = config.averageSlope() * (1.0 - config.concavityFactor());

                // Pendiente base estructural
                double currentBaseSlope = maxSlope - progress * (maxSlope - minSlope);
                double baseDrop = currentBaseSlope * dx;

                // Modulación por ruido (Sinuosidad vertical)
                double localSlopeVariability = config.slopeVariability() * macroMultiplier;
                double noiseEffectOnDrop = nDetail * localSlopeVariability;

                // Caída final y actualización de elevación
                double totalDrop = Math.max(0, baseDrop + noiseEffectOnDrop);
                elevationProfile[i] = (float) (elevationProfile[i - 1] - totalDrop);

                // La pendiente local resultante (Driver para modelos físicos)
                currentLocalSlope = totalDrop / dx;
            } else {
                // Para la celda 0, usamos la pendiente de cabecera estimada
                currentLocalSlope = config.averageSlope() * (1.0 + config.concavityFactor());
            }

            // --- C. Aplicación de Modelos Físicos (Delegación) ---

            // 1. ANCHO: Driver = Pendiente Local. Ruido = nMain (Sinuosidad del cauce)
            // El decorador añadirá automáticamente el ensanchamiento río abajo.
            bottomWidth[i] = widthModel.calculate(i, config, currentLocalSlope, nMain);

            // 2. MANNING: Driver = Pendiente Local. Ruido = nDetail (Obstáculos locales)
            manningCoefficient[i] = manningModel.calculate(i, config, currentLocalSlope, nDetail);

            // 3. DECAY: Driver = Manning (Turbulencia). Ruido = nDetail (Biomasa local)
            baseDecayCoefficientAt20C[i] = decayModel.calculate(i, config, manningCoefficient[i], nDetail);

            // 4. PH: Driver = Ruido Zonal (Tipo de Roca). Ruido = nDetail (Variación local)
            phProfile[i] = phModel.calculate(i, config, nZone, nDetail);

            // 5. Taludes (Side Slope): Driver = Manning. Con decorador de ruido zonal y detalle
            sideSlope[i] = sideSlopeModel.calculate(i, config, manningCoefficient[i], nDetail);

            // 6. Dispersión (Driver: Manning) -> Dependencia en Cascada
            dispersionAlpha[i] = dispersionModel.calculate(i, config, manningCoefficient[i], nDetail);
        }

        // 5. Construcción del Objeto Final
        return RiverGeometry.builder()
                .cellCount(cellCount)
                .spatialResolution(dx)
                .elevationProfile(elevationProfile)
                .bottomWidth(bottomWidth)
                .sideSlope(sideSlope)
                .manningCoefficient(manningCoefficient)
                .baseDecayCoefficientAt20C(baseDecayCoefficientAt20C)
                .phProfile(phProfile)
                .sectionTypes(sectionTypes)
                .dispersionAlpha(dispersionAlpha)
                .build();
    }

    /**
     * Aplica una lista de eventos geológicos en serie a una geometría de río base.
     */
    public static RiverGeometry applyGeologicalEvents(RiverGeometry baseRiver, List<GeologicalEvent> events) {
        if (events == null || events.isEmpty()) return baseRiver;
        validateEventSeparation(baseRiver.getSpatialResolution(), events);

        float[] newElevationProfile = baseRiver.cloneElevationProfile();
        float[] newBottomWidth = baseRiver.cloneBottomWidth();
        float[] newManningCoefficient = baseRiver.cloneManningCoefficient();
        RiverSectionType[] newSectionTypes = baseRiver.cloneSectionTypes();

        for (GeologicalEvent event : events) {
            event.apply(baseRiver.getSpatialResolution(), newElevationProfile, newBottomWidth, newManningCoefficient, newSectionTypes);
        }

        return baseRiver
                .withSectionTypes(newSectionTypes)
                .withElevationProfile(newElevationProfile)
                .withManningCoefficient(newManningCoefficient)
                .withBottomWidth(newBottomWidth);
    }

    private static void validateEventSeparation(double spatialResolution, List<GeologicalEvent> events) {
        if (events.size() <= 1) return;
        List<GeologicalEvent> sortedEvents = new ArrayList<>(events);
        sortedEvents.sort(Comparator.comparingDouble(GeologicalEvent::getPosition));
        int lastEventCell = (int) Math.round(events.get(0).getPosition() / spatialResolution);
        for (int i = 1; i < events.size(); i++) {
            int currentEventCell = (int) Math.round(events.get(i).getPosition() / spatialResolution);
            if (Math.abs(currentEventCell - lastEventCell) < MINIMUM_CELL_SEPARATION) {
                throw new IllegalArgumentException("Eventos geológicos demasiado juntos.");
            }
            lastEventCell = currentEventCell;
        }
    }
}