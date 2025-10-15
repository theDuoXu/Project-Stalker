package projectstalker.factory;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import projectstalker.config.RiverConfig;
import projectstalker.domain.river.InitialRiver;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;
import projectstalker.physics.impl.ManningHydrologySolver;
import projectstalker.physics.solver.IHydrologySolver;

import java.util.Arrays;
import java.util.DoubleSummaryStatistics;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Pruebas unitarias para la clase {@link RiverFactory}.
 * Verifica que la fábrica puede construir un estado de río inicial estable.
 */
class RiverFactoryTest {

    private RiverConfig config;
    private RiverFactory riverFactory;

    @BeforeEach
    void setUp() {
        // 1. Crear las dependencias necesarias para la RiverFactory
        RiverGeometryFactory geometryFactory = new RiverGeometryFactory();
        IHydrologySolver hydrologySolver = new ManningHydrologySolver();

        // 2. Instanciar la fábrica que vamos a probar, inyectando sus dependencias
        this.riverFactory = new RiverFactory(geometryFactory, hydrologySolver);

        // 3. Crear una configuración estándar para usar en el test
        this.config = new RiverConfig(
                12345L, 0.0f, 0.05f, 0.001f, 100000.0, 50.0, 200.0, 0.4, 0.0002,
                0.0001, 150.0, 40.0, 4.0, 1.5, 0.030, 0.005, 0.1, 0.05,
                15.0, 2.0, 8.0, 14.0, 7.5, 0.5,
                4.0, 20000.0, 1.5, 1.0, 0.25
        );
    }

    @Test
    @DisplayName("Debería crear un río estable y completamente poblado")
    void createStableRiver_shouldReturnPopulatedRiver() {
        // --- 1. Arrange ---
        final double initialDischarge = 150.0; // Caudal para estabilizar el río.

        // --- 2. Act ---
        // Llamamos al método que queremos probar
        InitialRiver initialRiver = riverFactory.createStableRiver(config, initialDischarge);

        // --- 3. Assert ---
        // Verificamos que el resultado es válido y coherente.
        assertNotNull(initialRiver, "El objeto InitialRiver no debería ser nulo.");
        assertNotNull(initialRiver.geometry(), "La geometría del río no debería ser nula.");
        assertNotNull(initialRiver.state(), "El estado del río no debería ser nulo.");

        RiverGeometry geometry = initialRiver.geometry();
        RiverState state = initialRiver.state();
        int lastCellIndex = geometry.getCellCount() - 1;

        // La aserción más importante: el agua debe haber llegado al final del río.
        assertTrue(state.getWaterDepthAt(lastCellIndex) > 0.01,
                "La profundidad en la última celda debe ser positiva, indicando que el río está lleno.");

        // Verificamos que el estado es consistente con la geometría.
        assertEquals(geometry.getCellCount(), state.waterDepth().length,
                "El número de celdas del estado y la geometría debe coincidir.");

        // Verificamos que el río ha alcanzado un estado aproximadamente estable.
        double outputDischarge = geometry.getCrossSectionalArea(lastCellIndex, state.getWaterDepthAt(lastCellIndex))
                * state.getVelocityAt(lastCellIndex);

        assertEquals(initialDischarge, outputDischarge, 5.0,
                "El caudal de salida debería ser aproximadamente igual al de entrada, indicando estabilidad.");

        System.out.println("Test de aserciones para RiverFactory superado con éxito.");

        // --- 4. Describe ---
        System.out.println("\n--- Resumen Estadístico del Río Estable Generado por la Factory ---");
        describeArray("Profundidad del Agua (m)", state.waterDepth());
        describeArray("Velocidad del Agua (m/s)", state.velocity());
        describeArray("Temperatura (°C)", state.temperature());
    }

    private void describeArray(String name, double[] data) {
        if (data == null || data.length == 0) {
            System.out.printf("\n--- %s ---\nDatos no disponibles o array vacío.\n", name);
            return;
        }
        System.out.printf("\n--- %s ---\n", name);
        DoubleSummaryStatistics stats = Arrays.stream(data).summaryStatistics();
        double mean = stats.getAverage();

        System.out.printf("Count:    %,d\n", stats.getCount());
        System.out.printf("Mean:     %.4f\n", mean);
        System.out.printf("Min:      %.4f\n", stats.getMin());
        System.out.printf("Max:      %.4f\n", stats.getMax());
        int headCount = Math.min(5, data.length);
        double[] head = Arrays.copyOfRange(data, 0, headCount);
        System.out.printf("Primeros %d valores: %s\n", headCount, Arrays.toString(head));
    }
}