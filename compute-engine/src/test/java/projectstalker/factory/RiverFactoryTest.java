package projectstalker.factory;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import projectstalker.config.RiverConfig;
import projectstalker.domain.river.InitialRiver;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;

import java.util.DoubleSummaryStatistics;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Pruebas unitarias para la clase {@link RiverFactory}.
 * Verifica que la fábrica puede construir un estado de río inicial estable
 * utilizando el método analítico (Newton-Raphson).
 */
class RiverFactoryTest {

    private static final Logger log = LoggerFactory.getLogger(RiverFactoryTest.class);

    private RiverConfig config;
    private RiverFactory riverFactory;

    @BeforeEach
    void setUp() {
        // 1. Crear las dependencias necesarias
        // Nota: Ya no necesitamos IHydrologySolver porque la fábrica usa cálculo directo.
        RiverGeometryFactory geometryFactory = new RiverGeometryFactory();

        // 2. Instanciar la fábrica
        this.riverFactory = new RiverFactory(geometryFactory);
        log.debug("RiverFactory inicializada (Modo Analítico).");

        // 3. Crear una configuración estándar
        this.config = RiverConfig.getTestingRiver();
    }

    @Test
    @DisplayName("Debería crear un río estacionario analíticamente correcto")
    void createStableRiver_shouldReturnSteadyStateRiver() {
        // --- 1. Arrange ---
        final float initialDischarge = 150.0f; // Caudal objetivo
        log.info("Iniciando test con caudal objetivo: {} m³/s", initialDischarge);

        // --- 2. Act ---
        // El método es ahora instantáneo (no hay bucle de simulación temporal)
        long t0 = System.nanoTime();
        InitialRiver initialRiver = riverFactory.createStableRiver(config, initialDischarge);
        long t1 = System.nanoTime();

        log.info("Río generado en {} µs", (t1 - t0) / 1000);

        // --- 3. Assert ---
        assertNotNull(initialRiver, "El objeto InitialRiver no debe ser nulo.");
        assertNotNull(initialRiver.geometry(), "La geometría no debe ser nula.");
        assertNotNull(initialRiver.state(), "El estado no debe ser nulo.");

        RiverGeometry geometry = initialRiver.geometry();
        RiverState state = initialRiver.state();
        int cellCount = geometry.getCellCount();

        // A. Verificación de Profundidad
        // El agua debe existir y ser mayor al mínimo técnico (0.001)
        assertTrue(state.getWaterDepthAt(0) > 0.001, "El inicio del río debe tener agua.");
        assertTrue(state.getWaterDepthAt(cellCount - 1) > 0.001, "El final del río debe tener agua.");

        // B. Verificación de Conservación de Masa (Steady State)
        // En un estado estacionario, el caudal Q debe ser idéntico en TODAS las celdas.
        // Q_calculado = V * A(h)
        float[] calculatedDischarges = state.discharge(geometry);

        double maxError = 0.0;
        for (int i = 0; i < cellCount; i++) {
            float qCell = calculatedDischarges[i];
            float error = Math.abs(qCell - initialDischarge);
            maxError = Math.max(maxError, error);

            // Tolerancia estricta: El método analítico es muy preciso.
            assertEquals(initialDischarge, qCell, 0.1f,
                    "Fallo de continuidad en celda " + i + ". El caudal no coincide con el objetivo.");
        }

        log.info("Validación de continuidad superada. Error máximo de caudal: {} m³/s", String.format("%.4f", maxError));

        // --- 4. Describe ---
        log.info("--- Resumen del Río Estacionario Generado ---");
        describeArray("Profundidad (m) [Varía según pendiente]", state.waterDepth());
        describeArray("Velocidad (m/s) [Varía según sección]", state.velocity());
        // Solo para verificar que calculamos Q correctamente (debería ser constante)
        describeArray("Caudal Calculado (m³/s) [Debe ser cte]", calculatedDischarges);
    }

    private void describeArray(String name, float[] data) {
        if (data == null || data.length == 0) return;

        DoubleSummaryStatistics stats = IntStream.range(0, data.length)
                .mapToDouble(i -> data[i])
                .summaryStatistics();

        log.info("{} -> Min: {}, Max: {}, Avg: {}",
                name,
                String.format("%.3f", stats.getMin()),
                String.format("%.3f", stats.getMax()),
                String.format("%.3f", stats.getAverage()));
    }
}