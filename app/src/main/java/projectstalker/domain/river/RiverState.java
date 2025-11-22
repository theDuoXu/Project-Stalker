package projectstalker.domain.river;

import lombok.Builder;
import lombok.With;

import java.util.Arrays;
import java.util.Objects;

/**
 * Representa una "instantánea" inmutable del estado físico-químico completo
 * del río en un único instante de tiempo 't'.
 * <p>
 * Como es un objeto de valor, dos instancias de {@code RiverState} se consideran
 * iguales si todos sus atributos correspondientes son iguales.
 *
 * @param waterDepth                Array con la profundidad del agua (H) en cada celda [m].
 * @param velocity                  Array con la velocidad del agua (v) en cada celda [m/s].
 * @param temperature               Array con la temperatura del agua (T) en cada celda [°C].
 * @param ph                        Array con el pH del agua en cada celda.
 * @param contaminantConcentration  Array con la concentración de contaminante en cada celda
 *
 * @author Duo Xu
 * @version 0.1
 * @since 2025-10-13
 */
@Builder
@With
public record RiverState(
        float[] waterDepth,
        float[] velocity,
        float[] temperature,
        float[] ph,
        float[] contaminantConcentration
) {
    /**
     * Constructor canónico que garantiza la validez e inmutabilidad del estado.
     * <p>
     * Realiza validaciones de nulidad y consistencia de dimensiones. Además, crea
     * "copias defensivas" de todos los arrays para asegurar que el estado interno
     * no pueda ser modificado externamente después de la creación del objeto.
     */
    public RiverState {
        // Validación de nulidad
        Objects.requireNonNull(waterDepth, "El array de profundidad de agua no puede ser nulo.");
        Objects.requireNonNull(velocity, "El array de velocidad no puede ser nulo.");
        Objects.requireNonNull(temperature, "El array de temperatura no puede ser nulo.");
        Objects.requireNonNull(ph, "El array de pH no puede ser nulo.");

        // Validación de consistencia de dimensiones
        int length = waterDepth.length;
        if (velocity.length != length || temperature.length != length || ph.length != length) {
            throw new IllegalArgumentException("Todos los arrays de estado deben tener la misma longitud.");
        }

        // Creación de copias defensivas para garantizar la inmutabilidad
        waterDepth = waterDepth.clone();
        velocity = velocity.clone();
        temperature = temperature.clone();
        ph = ph.clone();
    }

    private void validateCellIndex(int cellIndex) {
        if (cellIndex < 0 || cellIndex >= this.waterDepth.length) {
            throw new IndexOutOfBoundsException("El índice de celda " + cellIndex + " está fuera de los límites [0, " + (this.waterDepth.length - 1) + "].");
        }
    }

    /**
     * Devuelve la profundidad del agua en una celda específica del río.
     *
     * @param cellIndex El índice de la celda.
     * @return La profundidad del agua en metros [m].
     * @throws IndexOutOfBoundsException si el índice de celda es inválido.
     */
    public float getWaterDepthAt(int cellIndex) {
        validateCellIndex(cellIndex);
        return this.waterDepth[cellIndex];
    }

    /**
     * Devuelve la velocidad del agua en una celda específica del río.
     *
     * @param cellIndex El índice de la celda.
     * @return La velocidad del agua en metros por segundo [m/s].
     * @throws IndexOutOfBoundsException si el índice de celda es inválido.
     */
    public float getVelocityAt(int cellIndex) {
        validateCellIndex(cellIndex);
        return this.velocity[cellIndex];
    }

    /**
     * Devuelve la temperatura del agua en una celda específica del río.
     *
     * @param cellIndex El índice de la celda.
     * @return La temperatura en grados Celsius [°C].
     * @throws IndexOutOfBoundsException si el índice de celda es inválido.
     */
    public float getTemperatureAt(int cellIndex) {
        validateCellIndex(cellIndex);
        return this.temperature[cellIndex];
    }

    /**
     * Devuelve el pH del agua en una celda específica del río.
     *
     * @param cellIndex El índice de la celda.
     * @return El valor del pH.
     * @throws IndexOutOfBoundsException si el índice de celda es inválido.
     */
    public float getPhAt(int cellIndex) {
        validateCellIndex(cellIndex);
        return this.ph[cellIndex];
    }

    // Métodos equals y hashCode estándar para records con arrays.
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        RiverState that = (RiverState) o;
        return Arrays.equals(waterDepth, that.waterDepth) &&
                Arrays.equals(velocity, that.velocity) &&
                Arrays.equals(temperature, that.temperature) &&
                Arrays.equals(ph, that.ph);
    }

    @Override
    public int hashCode() {
        int result = Arrays.hashCode(waterDepth);
        result = 31 * result + Arrays.hashCode(velocity);
        result = 31 * result + Arrays.hashCode(temperature);
        result = 31 * result + Arrays.hashCode(ph);
        return result;
    }
}