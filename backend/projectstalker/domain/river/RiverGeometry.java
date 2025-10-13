package projectstalker.domain.river;

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
 * @version 1.0
 * @since 2025-10-12
 */

public final class RiverGeometry {

    private final int cellCount;
    private final double dx;
    private final double[] elevationProfile;
    private final double[] bottomWidth;
    private final double[] sideSlope;
    private final double[] manningCoefficient;
    private final double[] baseDecayCoefficientAt20C;

    /**
     * Constructor con visibilidad de paquete, diseñado para ser invocado
     * exclusivamente por {@code RiverGeometryFactory}.
     * <p>
     * Realiza una validación exhaustiva de los parámetros de entrada para asegurar
     * que el objeto RiverGeometry siempre se encuentre en un estado consistente y válido.
     *
     * @param cellCount          El número total de celdas del río (> 1).
     * @param dx                 La longitud de cada celda en metros (> 0).
     * @param elevationProfile   Array con la altitud del fondo del cauce para cada celda.
     * @param bottomWidth        Array con el ancho del fondo del cauce para cada celda (valores >= 0).
     * @param sideSlope          Array con la pendiente de los taludes laterales (valores >= 0).
     * @param manningCoefficient Array con el coeficiente de rugosidad de Manning (valores > 0).
     */
    RiverGeometry(int cellCount, double dx, double[] elevationProfile, double[] bottomWidth, double[] sideSlope,
                  double[] manningCoefficient, double[] baseDecayCoefficientAt20C) {
        // --- Validación de Parámetros ---
        if (cellCount <= 1) {
            throw new IllegalArgumentException("El número de celdas debe ser mayor que 1.");
        }
        if (dx <= 0) {
            throw new IllegalArgumentException("La resolución espacial (dx) debe ser positiva.");
        }

        // Validar que los arrays no sean nulos
        Objects.requireNonNull(elevationProfile, "El perfil de elevación no puede ser nulo.");
        Objects.requireNonNull(bottomWidth, "El array de ancho de fondo no puede ser nulo.");
        Objects.requireNonNull(sideSlope, "El array de pendiente de taludes no puede ser nulo.");
        Objects.requireNonNull(manningCoefficient, "El array del coeficiente de Manning no puede ser nulo.");
        Objects.requireNonNull(baseDecayCoefficientAt20C, "El array del coeficiente de decaída no puede ser nulo.");

        // Validar que todos los arrays tengan la longitud correcta
        if (elevationProfile.length != cellCount || bottomWidth.length != cellCount ||
                sideSlope.length != cellCount || manningCoefficient.length != cellCount || baseDecayCoefficientAt20C.length != cellCount) {
            throw new IllegalArgumentException("Todos los arrays de propiedades deben tener una longitud igual a cellCount.");
        }


        // Asegurar que el río siempre fluya cuesta abajo (perfil monotónicamente no creciente) ---
        for (int i = 0; i < cellCount - 1; i++) {
            if (elevationProfile[i] < elevationProfile[i + 1]) {
                throw new IllegalArgumentException(
                        String.format("El perfil de elevación es físicamente inconsistente. La altitud aumenta de la celda %d (%.2fm) a la celda %d (%.2fm)." +
                                        " debe ser monotónicamente no creciente",
                                i, elevationProfile[i], i + 1, elevationProfile[i + 1])
                );
            }
        }

        // Clonamos los arrays para garantizar una inmutabilidad completa.
        // Esto previene que el código externo modifique el estado interno de esta clase
        // si mantiene una referencia a los arrays originales.
        this.cellCount = cellCount;
        this.dx = dx;
        this.elevationProfile = elevationProfile.clone();
        this.bottomWidth = bottomWidth.clone();
        this.sideSlope = sideSlope.clone();
        this.manningCoefficient = manningCoefficient.clone();
        this.baseDecayCoefficientAt20C = baseDecayCoefficientAt20C.clone();
    }

    // --- MÉTODOS DE CONSULTA ESTÁTICA ---

    public int getCellCount() {
        return cellCount;
    }

    public double getDx() {
        return dx;
    }

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
    public double getBedSlopeAt(int cellIndex) {
        validateCellIndex(cellIndex);
        int nextIndex = (cellIndex == cellCount - 1) ? cellIndex - 1 : cellIndex;
        return (elevationProfile[nextIndex] - elevationProfile[nextIndex + 1]) / dx;
    }

    public double getManningAt(int cellIndex) {
        validateCellIndex(cellIndex);
        return manningCoefficient[cellIndex];
    }

    public double getBaseDecayAt(int cellIndex) {
        validateCellIndex(cellIndex);
        return manningCoefficient[cellIndex];
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
     * Valida que el índice de la celda proporcionado esté dentro del rango válido.
     * El rango válido es [0, cellCount - 1].
     *
     * @param cellIndex El índice de la celda a validar.
     * @throws IndexOutOfBoundsException si el índice está fuera del rango permitido.
     */
    private void validateCellIndex(int cellIndex) {
        if (cellIndex < 0 || cellIndex >= cellCount) {
            throw new IndexOutOfBoundsException(
                    String.format(
                            "Índice de celda fuera de rango: %d. El rango válido es de 0 a %d.",
                            cellIndex,
                            cellCount - 1
                    )
            );
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
            throw new IllegalArgumentException(
                    "La profundidad del agua no puede ser negativa: " + waterDepth
            );
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
        double totalLengthKm = (cellCount * dx) / 1000.0;
        return String.format(
                "RiverGeometry {\n" +
                        "  cellCount=%d,\n" +
                        "  dx=%.2f m,\n" +
                        "  totalLength=%.2f km,\n" +
                        "  elevationProfile=[%.2f m ... %.2f m],\n" +
                        "  bottomWidth=[%.2f m ... %.2f m],\n" +
                        "  sideSlope=[%.2f ... %.2f],\n" +
                        "  manningCoefficient=[%.3f ... %.3f]\n" +
                        "  decayCoefficient=[%.3f ... %.3f]\n"+
                "}",
                cellCount,
                dx,
                totalLengthKm,
                elevationProfile[0], elevationProfile[cellCount - 1],
                bottomWidth[0], bottomWidth[cellCount - 1],
                sideSlope[0], sideSlope[cellCount - 1],
                manningCoefficient[0], manningCoefficient[cellCount - 1],
                baseDecayCoefficientAt20C[0], baseDecayCoefficientAt20C[cellCount - 1]
        );
    }


    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        RiverGeometry that = (RiverGeometry) o;
        return cellCount == that.cellCount &&
                Double.compare(that.dx, dx) == 0 &&
                Arrays.equals(elevationProfile, that.elevationProfile) &&
                Arrays.equals(bottomWidth, that.bottomWidth) &&
                Arrays.equals(sideSlope, that.sideSlope) &&
                Arrays.equals(manningCoefficient, that.manningCoefficient) &&
                Arrays.equals(baseDecayCoefficientAt20C, that.baseDecayCoefficientAt20C);
    }

    @Override
    /**
     * Implementación para evitar colisiones y respetar orden de los atributos
     */
    public int hashCode() {
        int result = Objects.hash(cellCount, dx);
        result = 31 * result + Arrays.hashCode(elevationProfile);
        result = 31 * result + Arrays.hashCode(bottomWidth);
        result = 31 * result + Arrays.hashCode(sideSlope);
        result = 31 * result + Arrays.hashCode(manningCoefficient);
        result = 31 * result + Arrays.hashCode(baseDecayCoefficientAt20C);
        return result;
    }
}