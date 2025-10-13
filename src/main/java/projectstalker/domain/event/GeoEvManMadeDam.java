package projectstalker.domain.event;

import projectstalker.domain.river.RiverSectionType;

/**
 * Simula la construcción de una presa artificial y la consecuente y gradual
 * sedimentación en el embalse a lo largo del tiempo.
 * <p>
 * El impacto de este evento depende de la edad de la presa (`damAge`):
 * <ol>
 * <li>Crea una barrera física e inalterable en la celda de la presa.</li>
 * <li>Modela la acumulación de sedimentos y el ensanchamiento del cauce aguas arriba
 * de forma proporcional a la edad de la presa, interpolando entre el lecho
 * original (presa nueva) y un perfil de colmatación total (presa antigua).</li>
 * </ol>
 */
public final class GeoEvManMadeDam implements GeologicalEvent {

    /**
     * Define la diferencia de altura entre la coronación de la presa y el nivel
     * máximo de sedimentación. Simula la altura del aliviadero principal.
     */
    private static final double SPILLWAY_ELEVATION_OFFSET = 1.0; // en metros

    /**
     * Define el número de años necesarios para que el embalse alcance
     * su nivel máximo de colmatación por sedimentos.
     */
    private static final double TIME_TO_FULL_SEDIMENTATION = 100.0; // en años

    private final double position;
    private final double crestElevation;
    private final double reservoirWidth;
    private final int damAge;

    /**
     * Constructor para un evento de presa artificial.
     *
     * @param position       La localización en metros donde se construirá la presa.
     * @param crestElevation La altitud máxima de la estructura de la presa en metros.
     * @param reservoirWidth El ancho en metros que alcanzará el embalse en su madurez.
     * @param damAge         La edad de la presa en años, para simular la sedimentación parcial.
     */
    public GeoEvManMadeDam(double position, double crestElevation, double reservoirWidth, int damAge) {
        if (position < 0 || crestElevation < 0 || reservoirWidth <= 0 || damAge < 0) {
            throw new IllegalArgumentException("Los parámetros de la presa deben ser positivos (la edad puede ser 0).");
        }
        this.position = position;
        this.crestElevation = crestElevation;
        this.reservoirWidth = reservoirWidth;
        this.damAge = damAge;
    }

    @Override
    public void apply(
            double spatialResolution,
            double[] elevationProfile,
            double[] bottomWidth,
            double[] manningCoefficient,
            RiverSectionType[] sectionTypes
    ) {
        final int cellCount = elevationProfile.length;
        final int damCell = (int) Math.round(position / spatialResolution);

        if (damCell <= 0 || damCell >= cellCount - 1) {
            return; // Ignorar si la presa está en los extremos del río.
        }

        // --- 1. Crear la barrera física de la presa (independiente de la edad) ---
        // Esto lo permite la excepción escrita en RiverGeometry
        sectionTypes[damCell] = RiverSectionType.DAM_STRUCTURE;
        elevationProfile[damCell] = this.crestElevation;
        bottomWidth[damCell] = 10.0; // Ancho de la estructura
        manningCoefficient[damCell] = 0.013; // Rugosidad del hormigón

        // --- 2. Calcular el factor de envejecimiento para la sedimentación ---
        final double ageFactor = Math.min(1.0, this.damAge / TIME_TO_FULL_SEDIMENTATION);
        if (ageFactor == 0.0) {
            return; // Presa nueva, no hay sedimentación que simular.
        }

        // --- 3. Calcular el perfil de sedimentación MÁXIMO (objetivo a 100 años) ---
        final double targetSedimentElevation = this.crestElevation - SPILLWAY_ELEVATION_OFFSET;
        final double maxSedimentDepthAtDam = targetSedimentElevation - elevationProfile[damCell - 1];

        if (maxSedimentDepthAtDam <= 0) {
            return; // No hay espacio vertical para la sedimentación.
        }

        int reservoirEndCell = 0;
        for (int i = damCell - 1; i >= 0; i--) {
            if (elevationProfile[i] >= targetSedimentElevation) {
                reservoirEndCell = i;
                break;
            }
        }

        final int sedimentationLengthInCells = (damCell - 1) - reservoirEndCell;
        if (sedimentationLengthInCells <= 0) {
            return;
        }

        // --- 4. Aplicar la sedimentación y ensanchamiento PARCIAL basado en ageFactor ---
        for (int i = damCell - 1; i >= reservoirEndCell; i--) {
            double progress = (double) (i - reservoirEndCell) / (double) sedimentationLengthInCells;

            // a. Interpolar la elevación
            double maxPossibleSedimentForCell = maxSedimentDepthAtDam * progress;
            double actualSedimentToAdd = maxPossibleSedimentForCell * ageFactor;
            elevationProfile[i] += actualSedimentToAdd;

            // b. Interpolar la anchura
            double originalWidth = bottomWidth[i];
            double maxWidthChange = this.reservoirWidth - originalWidth;
            double actualWidthToAdd = maxWidthChange * progress * ageFactor;
            bottomWidth[i] += actualWidthToAdd;

            // c. Actualizar propiedades del lecho (se consideran de embalse en cuanto hay sedimento)
            manningCoefficient[i] = 0.025;
            sectionTypes[i] = RiverSectionType.RESERVOIR;
        }
    }
}