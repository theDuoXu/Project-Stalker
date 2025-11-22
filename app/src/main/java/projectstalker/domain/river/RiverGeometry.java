package projectstalker.domain.river;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Builder;
import lombok.Getter;
import lombok.With;

import java.util.Objects;
import java.util.Arrays;

/**
 * Representa el plano maestro inmutable de la morfología de un río.
 * <p>
 * Esta clase es la fuente única de la verdad sobre la forma física y estática
 * del cauce. Una vez instanciada, no puede ser modificada, garantizando la
 * seguridad en entornos concurrentes. Provee métodos para calcular propiedades
 * hidráulicas que dependen de un contexto dinámico (como la altura del agua),
 * el cual siempre es proporcionado como un argumento externo.
 *
 * @author Duo Xu
 * @version 0.1
 * @since 2025-10-12
 */
@Builder
@With
public final class RiverGeometry {

    @Getter
    private final int cellCount;
    @Getter
    private final float spatial_resolution;
    private final float[] elevationProfile;
    private final float[] bottomWidth;
    private final float[] sideSlope;
    private final float[] manningCoefficient;
    private final float[] baseDecayCoefficientAt20C;
    private final float[] phProfile;
    private final float[] dispersionAlpha;
    private final RiverSectionType[] sectionTypes;

    /**
     * Constructor diseñado para ser invocado
     * exclusivamente por {@code RiverGeometryFactory}.
     * <p>
     * Realiza una validación exhaustiva de los parámetros de entrada para asegurar
     * que el objeto RiverGeometry siempre se encuentre en un estado consistente y válido.
     *
     * @param cellCount          El número total de celdas del río (> 1).
     * @param spatial_resolution                 La longitud de cada celda en metros (> 0).
     * @param elevationProfile   Array con la altitud del fondo del cauce para cada celda.
     * @param bottomWidth        Array con el ancho del fondo del cauce para cada celda (valores >= 0).
     * @param sideSlope          Array con la pendiente de los taludes laterales (valores >= 0).
     * @param manningCoefficient Array con el coeficiente de rugosidad de Manning (valores > 0).
     * @param phProfile          Array con el perfil PH del río
     * @param dispersionAlpha    Coeficiente alpha para Taylor
     * @param sectionTypes       Tipos de eventos geológicos del río
     */
    @JsonCreator
    public RiverGeometry(@JsonProperty("cellCount") int cellCount,
                         @JsonProperty("dx") float spatial_resolution,
                         @JsonProperty("elevationProfile") float[] elevationProfile,
                         @JsonProperty("bottomWidth") float[] bottomWidth,
                         @JsonProperty("sideSlope") float[] sideSlope,
                         @JsonProperty("manningCoefficient") float[] manningCoefficient,
                         @JsonProperty("baseDecayCoefficientAt20C") float[] baseDecayCoefficientAt20C,
                         @JsonProperty("phProfile") float[] phProfile,
                         @JsonProperty("dispersionAlpha") float[] dispersionAlpha,
                         @JsonProperty("sectionTypes") RiverSectionType[] sectionTypes) {
        // --- Validación de Parámetros ---
        if (cellCount <= 1) {
            throw new IllegalArgumentException("El número de celdas debe ser mayor que 1.");
        }
        if (spatial_resolution <= 0) {
            throw new IllegalArgumentException("La resolución espacial (dx) debe ser positiva.");
        }

        // Validar que los arrays no sean nulos
        Objects.requireNonNull(elevationProfile, "El perfil de elevación no puede ser nulo.");
        Objects.requireNonNull(bottomWidth, "El array de ancho de fondo no puede ser nulo.");
        Objects.requireNonNull(sideSlope, "El array de pendiente de taludes no puede ser nulo.");
        Objects.requireNonNull(manningCoefficient, "El array del coeficiente de Manning no puede ser nulo.");
        Objects.requireNonNull(baseDecayCoefficientAt20C, "El array del coeficiente de decaída no puede ser nulo.");
        Objects.requireNonNull(phProfile, "El array del perfil de ph no puede ser nulo.");
        Objects.requireNonNull(sectionTypes, "El array del tipo de celda del río no puede ser nulo.");
        Objects.requireNonNull(dispersionAlpha, "El array de coeficiente de dispersión (alpha) no puede ser nulo.");

        // Validar que todos los arrays tengan la longitud correcta
        if (elevationProfile.length != cellCount || bottomWidth.length != cellCount || sideSlope.length != cellCount || manningCoefficient.length != cellCount || baseDecayCoefficientAt20C.length != cellCount || phProfile.length != cellCount || sectionTypes.length != cellCount || dispersionAlpha.length != cellCount) {
            throw new IllegalArgumentException("Todos los arrays de propiedades deben tener una longitud igual a cellCount.");
        }


        // --- Asegurar que el perfil sea físicamente consistente ---
        // El lecho debe ser monotónicamente no creciente, con una única excepción:
        // se permite un "salto" hacia arriba justo antes de una estructura de presa.
        for (int i = 0; i < cellCount - 1; i++) {
            if (elevationProfile[i] < elevationProfile[i + 1]) {
                // Se ha detectado un aumento en la elevación. Comprobamos si es la excepción permitida.
                boolean isDamException = (sectionTypes != null && sectionTypes[i + 1] == RiverSectionType.DAM_STRUCTURE);

                if (!isDamException) {
                    // Si no es la excepción de la presa, el perfil es inválido.
                    throw new IllegalArgumentException(String.format("El perfil de elevación es físicamente inconsistente. La altitud aumenta de la celda %d (%.2fm) a la celda %d (%.2fm) " + "sin que la siguiente celda sea una presa.", i, elevationProfile[i], i + 1, elevationProfile[i + 1]));
                }
            }
        }

        // Clonamos los arrays para garantizar una inmutabilidad completa.
        // Esto previene que el código externo modifique el estado interno de esta clase
        // si mantiene una referencia a los arrays originales.
        this.cellCount = cellCount;
        this.spatial_resolution = spatial_resolution;
        this.elevationProfile = elevationProfile.clone();
        this.bottomWidth = bottomWidth.clone();
        this.sideSlope = sideSlope.clone();
        this.manningCoefficient = manningCoefficient.clone();
        this.baseDecayCoefficientAt20C = baseDecayCoefficientAt20C.clone();
        this.phProfile = phProfile.clone();
        this.dispersionAlpha = dispersionAlpha.clone();
        this.sectionTypes = sectionTypes.clone();
    }

    // --- MÉTODOS DE CONSULTA ESTÁTICA ---

    /**
     * Calcula la pendiente del fondo del cauce en una celda específica.
     * La pendiente se calcula como la diferencia de altitud con la celda siguiente
     * dividida por la distancia (dx). Para la última celda, se asume la misma
     * pendiente que la penúltima para evitar errores de contorno.
     *
     * @param cellIndex El índice de la celda (0 a N-1).
     * @return La pendiente del cauce (adimensional, m/m).
     * @throws IndexOutOfBoundsException si el índice está fuera de rango.
     */
    public float getBedSlopeAt(int cellIndex) {
        validateCellIndex(cellIndex);
        int nextIndex = (cellIndex == cellCount - 1) ? cellIndex - 1 : cellIndex;
        return (elevationProfile[nextIndex] - elevationProfile[nextIndex + 1]) / spatial_resolution;
    }

    public float getManningAt(int cellIndex) {
        validateCellIndex(cellIndex);
        return manningCoefficient[cellIndex];
    }

    public float getBaseDecayAt(int cellIndex) {
        validateCellIndex(cellIndex);
        return baseDecayCoefficientAt20C[cellIndex];
    }

    public float getWidthAt(int cellIndex) {
        validateCellIndex(cellIndex);
        return bottomWidth[cellIndex];
    }

    /**
     * Devuelve el pH base del agua para una celda específica.
     *
     * @param cellIndex El índice de la celda.
     * @return El valor del pH para esa sección del río.
     */
    public float getPhAt(int cellIndex) {
        validateCellIndex(cellIndex);
        return phProfile[cellIndex];
    }


    // --- MÉTODOS DE CONSULTA HIDRÁULICA (REQUIEREN CONTEXTO DINÁMICO) ---

    /**
     * Calcula el área de la sección transversal mojada para una celda.
     * La sección se modela como un trapecio.
     *
     * @param cellIndex  El índice de la celda.
     * @param waterDepth La profundidad del agua (H) en metros (>= 0).
     * @return El área de la sección mojada en metros cuadrados (m²).
     */
    public double getCrossSectionalArea(int cellIndex, double waterDepth) {
        validateCellIndex(cellIndex);
        validateWaterDepth(waterDepth);

        double b = bottomWidth[cellIndex];
        double m = sideSlope[cellIndex];
        return (b + m * waterDepth) * waterDepth;
    }

    /**
     * Calcula el perímetro mojado para una celda.
     *
     * @param cellIndex  El índice de la celda.
     * @param waterDepth La profundidad del agua (H) en metros (>= 0).
     * @return El perímetro mojado en metros (m).
     */
    public double getWettedPerimeter(int cellIndex, double waterDepth) {
        validateCellIndex(cellIndex);
        validateWaterDepth(waterDepth);

        double b = bottomWidth[cellIndex];
        double m = sideSlope[cellIndex];
        return b + 2.0 * waterDepth * Math.sqrt(1.0 + m * m);
    }

    /**
     * Calcula el radio hidráulico para una celda.
     * El radio hidráulico es una medida de la eficiencia del flujo del canal.
     *
     * @param cellIndex  El índice de la celda.
     * @param waterDepth La profundidad del agua (H) en metros (>= 0).
     * @return El radio hidráulico en metros (m).
     */
    public double getHydraulicRadius(int cellIndex, double waterDepth) {
        // La validación de los parámetros la realizan los métodos internos
        double area = getCrossSectionalArea(cellIndex, waterDepth);
        double perimeter = getWettedPerimeter(cellIndex, waterDepth);

        if (perimeter == 0.0) {
            return 0.0;
        }
        return area / perimeter;
    }

    /**
     * Devuelve el coeficiente de proporcionalidad de dispersión (alpha).
     * Usado para calcular D_L = alpha * H * u
     */
    public float getDispersionAlphaAt(int cellIndex) {
        validateCellIndex(cellIndex);
        return dispersionAlpha[cellIndex];
    }

    public float getSideSlopeAt(int cellIndex) {
        validateCellIndex(cellIndex);
        return sideSlope[cellIndex];
    }

    /**
     * Devuelve el tipo de sección para una celda específica.
     *
     * @param cellIndex El índice de la celda.
     * @return El {@link RiverSectionType} de esa sección del río.
     */
    public RiverSectionType getSectionTypeAt(int cellIndex) {
        validateCellIndex(cellIndex);
        return sectionTypes[cellIndex];
    }

    /**
     * Valida que el índice de la celda proporcionado esté dentro del rango válido.
     * El rango válido es [0, cellCount - 1].
     *
     * @param cellIndex El índice de la celda a validar.
     * @throws IndexOutOfBoundsException si el índice está fuera del rango permitido.
     */
    private void validateCellIndex(int cellIndex) {
        if (cellIndex < 0 || cellIndex >= cellCount) {
            throw new IndexOutOfBoundsException(String.format("Índice de celda fuera de rango: %d. El rango válido es de 0 a %d.", cellIndex, cellCount - 1));
        }
    }

    /**
     * Valida que la profundidad del agua proporcionada sea físicamente plausible.
     * La profundidad del agua no puede ser un valor negativo.
     *
     * @param waterDepth La profundidad del agua en metros a validar.
     * @throws IllegalArgumentException si la profundidad del agua es negativa.
     */
    private void validateWaterDepth(double waterDepth) {
        if (waterDepth < 0) {
            throw new IllegalArgumentException("La profundidad del agua no puede ser negativa: " + waterDepth);
        }
    }

    /**
     * Calcula el ancho de la superficie libre del agua para una celda. *
     * <p>
     * <p>
     * Este valor es importante para visualizaciones, cálculos de evaporación o
     * modelos de dispersión más avanzados. Se calcula a partir de la geometría
     * trapezoidal del cauce.
     *
     * @param cellIndex  El índice de la celda.
     * @param waterDepth La profundidad del agua (H) en metros (>= 0).
     * @return El ancho de la superficie del agua (T) en metros (m).
     */
    public double getTopWidth(int cellIndex, double waterDepth) {
        validateCellIndex(cellIndex);
        validateWaterDepth(waterDepth);

        double b = bottomWidth[cellIndex];
        double m = sideSlope[cellIndex];

        // Fórmula: Ancho del fondo + 2 * (proyección horizontal del talud mojado)
        return b + 2.0 * m * waterDepth;
    }

    /**
     * Devuelve una representación en formato de cadena de texto del objeto RiverGeometry.
     * <p>
     * Es útil para la depuración y el registro (logging), proporcionando un resumen
     * claro de las propiedades clave de la geometría del río.
     *
     * @return Una cadena de texto que describe el objeto.
     */
    @Override
    public String toString() {
        double totalLengthKm = (cellCount * spatial_resolution) / 1000.0;
        return String.format("RiverGeometry {\n" + "  cellCount=%d,\n" + "  dx=%.2f m,\n" + "  totalLength=%.2f km,\n" + "  elevationProfile=[%.2f m ... %.2f m],\n" + "  bottomWidth=[%.2f m ... %.2f m],\n" + "  sideSlope=[%.2f ... %.2f],\n" + "  manningCoefficient=[%.3f ... %.3f]\n" + "  decayCoefficient=[%.3f ... %.3f]\n" + "  this river has %d types of cells\n" + "}", cellCount, spatial_resolution, totalLengthKm, elevationProfile[0], elevationProfile[cellCount - 1], bottomWidth[0], bottomWidth[cellCount - 1], sideSlope[0], sideSlope[cellCount - 1], manningCoefficient[0], manningCoefficient[cellCount - 1], baseDecayCoefficientAt20C[0], baseDecayCoefficientAt20C[cellCount - 1], Arrays.stream(sectionTypes).distinct().count());
    }


    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        RiverGeometry that = (RiverGeometry) o;
        return cellCount == that.cellCount && Double.compare(that.spatial_resolution, spatial_resolution) == 0 && Arrays.equals(elevationProfile, that.elevationProfile) && Arrays.equals(bottomWidth, that.bottomWidth) && Arrays.equals(sideSlope, that.sideSlope) && Arrays.equals(manningCoefficient, that.manningCoefficient) && Arrays.equals(baseDecayCoefficientAt20C, that.baseDecayCoefficientAt20C) && Arrays.equals(phProfile, that.phProfile) && Arrays.equals(sectionTypes, that.sectionTypes);
    }

    @Override
    /**
     * Implementación para evitar colisiones y respetar orden de los atributos
     */ public int hashCode() {
        int result = Objects.hash(cellCount, spatial_resolution);
        result = 31 * result + Arrays.hashCode(elevationProfile);
        result = 31 * result + Arrays.hashCode(bottomWidth);
        result = 31 * result + Arrays.hashCode(sideSlope);
        result = 31 * result + Arrays.hashCode(manningCoefficient);
        result = 31 * result + Arrays.hashCode(baseDecayCoefficientAt20C);
        result = 31 * result + Arrays.hashCode(phProfile);
        result = 31 * result + Arrays.hashCode(sectionTypes);
        return result;
    }

    public float[] cloneElevationProfile() {
        return elevationProfile.clone();
    }

    public float[] cloneBottomWidth() {
        return bottomWidth.clone();
    }

    public float[] cloneSideSlope() {
        return sideSlope.clone();
    }

    public float[] cloneManningCoefficient() {
        return manningCoefficient.clone();
    }

    public float[] cloneBaseDecayCoefficientAt20C() {
        return baseDecayCoefficientAt20C.clone();
    }

    public float[] clonePhProfile() {
        return phProfile.clone();
    }

    public RiverSectionType[] cloneSectionTypes() {
        return sectionTypes.clone();
    }
}