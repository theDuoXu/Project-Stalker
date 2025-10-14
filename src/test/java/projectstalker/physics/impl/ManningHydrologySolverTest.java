package projectstalker.physics.impl;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import projectstalker.config.RiverConfig;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;
import projectstalker.factory.RiverGeometryFactory;

import java.util.Arrays;
import java.util.DoubleSummaryStatistics;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Pruebas unitarias para la clase {@link ManningHydrologySolver}.
 * Verifica que el solver calcula correctamente el siguiente estado hidrológico del río.
 */
class ManningHydrologySolverTest {

    private RiverConfig config;
    private RiverGeometry riverGeometry;
    private ManningHydrologySolver solver;

    @BeforeEach
    void setUp() {
        // Usamos la configuración realista para un río tipo Tajo.
        config = new RiverConfig(
                12345L, 0.0f, 0.05f, 0.001f, 100000.0, 50.0, 200.0, 0.4, 0.0002,
                0.0001, 150.0, 40.0, 4.0, 1.5, 0.030, 0.005, 0.1, 0.05,
                15.0, 2.0, 8.0, 14.0, 7.5, 0.5,
                4.0, 20000.0, 1.5, 1.0, 0.25
        );
        RiverGeometryFactory factory = new RiverGeometryFactory();
        riverGeometry = factory.createRealisticRiver(config);
        solver = new ManningHydrologySolver();
    }

    @Test
    @DisplayName("Debería calcular el primer estado del río a partir de un cauce seco")
    void calculateNextState_fromDryBed_shouldProduceFlow() {
        // --- 1. Arrange ---
        final double inputDischarge = 150.0;
        final double currentTimeInSeconds = 0.0;
        final int cellCount = riverGeometry.getCellCount();
        RiverState initialState = new RiverState(
                new double[cellCount], new double[cellCount], new double[cellCount], new double[cellCount]
        );

        // --- 2. Act ---
        RiverState nextState = solver.calculateNextState(
                initialState, riverGeometry, config, currentTimeInSeconds, inputDischarge
        );

        // --- 3. Assert ---
        assertNotNull(nextState, "El estado calculado no debería ser nulo.");
        assertTrue(nextState.getWaterDepthAt(0) > 0.1, "La profundidad en la primera celda debe ser positiva.");
        assertTrue(nextState.getVelocityAt(0) > 0.1, "La velocidad en la primera celda debe ser positiva.");

        double firstCellArea = riverGeometry.getCrossSectionalArea(0, nextState.getWaterDepthAt(0));
        double firstCellDischarge = firstCellArea * nextState.getVelocityAt(0);
        assertEquals(inputDischarge, firstCellDischarge, 0.01, "El caudal en la primera celda debe ser igual al de entrada.");

        // Verificamos que el modelo espacial de temperatura funciona
        // En t=0, la temperatura base es la media anual.
        double baseTemp = config.averageAnnualTemperature();
        double tempAtHeadwater = nextState.getTemperatureAt(0);
        // La temperatura en la cabecera (celda 0) debe ser significativamente más fría que la base
        // debido al 'maxHeadwaterCoolingEffect'.
        assertTrue(tempAtHeadwater < baseTemp, "La temperatura en la cabecera debería ser más fría que la base.");
        // Comprobamos que es aproximadamente igual a la base menos el efecto de enfriamiento (con una tolerancia para otros efectos).
        assertEquals(baseTemp - config.maxHeadwaterCoolingEffect(), tempAtHeadwater, 1.0, "La temperatura en la cabecera no refleja el efecto de enfriamiento esperado.");

        assertEquals(riverGeometry.getPhAt(50), nextState.getPhAt(50), 1e-6);
        assertEquals(0.0, nextState.getWaterDepthAt(cellCount - 1), "La última celda aún debería estar seca.");

        System.out.println("Test 'calculateNextState_fromDryBed' superado. El solver funciona de manera coherente.");

        // --- 4. Describe ---
        System.out.println("\n--- Resumen Estadístico del Primer Estado del Río (t=1) ---");
        describeArray("Profundidad del Agua (m)", nextState.waterDepth());
        describeArray("Velocidad del Agua (m/s)", nextState.velocity());
        describeArray("Temperatura (°C)", nextState.temperature());
        describeArray("pH", nextState.ph());
    }

    @Test
    @DisplayName("Debería simular el llenado hasta estado estacionario y luego una crecida")
    void simulateSteadyStateAndFloodWave() {
        // --- 0. Constantes y Configuración Inicial ---
        final int cellCount = riverGeometry.getCellCount();
        final int lastCellIndex = cellCount - 1;
        final double steadyDischarge = 150.0;
        final double floodDischarge = 200.0;
        final int maxIterations = cellCount * 2;

        // --- FASE 1: Llenado del río hasta alcanzar el estado estacionario ---
        System.out.println("--- FASE 1: Llenando el río con 150 m³/s hasta el estado estacionario... ---");
        RiverState currentState = new RiverState(new double[cellCount], new double[cellCount], new double[cellCount], new double[cellCount]);
        long timeInSeconds = 0;
        int iterations = 0;

        while (currentState.getWaterDepthAt(lastCellIndex) < 0.01) {
            currentState = solver.calculateNextState(currentState, riverGeometry, config, timeInSeconds, steadyDischarge);
            timeInSeconds += 3600; // Avanzamos 1 hora por paso para que la temperatura evolucione
            iterations++;
            if (iterations > maxIterations) {
                fail("El llenado del río superó el máximo de " + maxIterations + " iteraciones.");
            }
        }

        System.out.printf("Río estabilizado después de %d iteraciones.\n", iterations);
        RiverState steadyState = currentState;

        // --- Análisis del Estado Estacionario ---
        System.out.println("\n--- Resumen Estadístico del Estado Estacionario (150 m³/s) ---");
        DoubleSummaryStatistics steadyStats = Arrays.stream(steadyState.waterDepth()).summaryStatistics();
        describeArray("Profundidad (Estado Estacionario)", steadyState.waterDepth());
        describeArray("Velocidad (Estado Estacionario)", steadyState.velocity());

        double steadyOutputDischarge = riverGeometry.getCrossSectionalArea(lastCellIndex, steadyState.getWaterDepthAt(lastCellIndex)) * steadyState.getVelocityAt(lastCellIndex);
        assertEquals(steadyDischarge, steadyOutputDischarge, 5.0, "El caudal de salida en estado estacionario debería ser cercano a 150 m³/s.");

        // --- FASE 2: Propagación de una crecida con 200 m³/s ---
        System.out.println("\n--- FASE 2: Introduciendo una crecida de 200 m³/s... ---");
        iterations = 0;
        double currentOutputDischarge = steadyOutputDischarge;

        while (Math.abs(currentOutputDischarge - floodDischarge) > 1.0) {
            currentState = solver.calculateNextState(currentState, riverGeometry, config, timeInSeconds, floodDischarge);
            currentOutputDischarge = riverGeometry.getCrossSectionalArea(lastCellIndex, currentState.getWaterDepthAt(lastCellIndex)) * currentState.getVelocityAt(lastCellIndex);
            timeInSeconds += 3600;
            iterations++;
            if (iterations > maxIterations) {
                fail("La simulación de la crecida superó el máximo de " + maxIterations + " iteraciones.");
            }
        }

        System.out.printf("Crecida estabilizada después de %d iteraciones adicionales.\n", iterations);
        RiverState floodState = currentState;

        // --- Análisis del Estado de la Crecida y Comparación ---
        System.out.println("\n--- Resumen Estadístico del Estado de Crecida (200 m³/s) ---");
        DoubleSummaryStatistics floodStats = Arrays.stream(floodState.waterDepth()).summaryStatistics();
        describeArray("Profundidad (Estado de Crecida)", floodState.waterDepth());
        describeArray("Velocidad (Estado de Crecida)", floodState.velocity());

        assertEquals(floodDischarge, currentOutputDischarge, 1.0, "El caudal de salida final debería ser cercano a 200 m³/s.");

        System.out.printf("\nComparación de Profundidad Media:\n");
        System.out.printf(" - Estado Estacionario (150 m³/s): %.4f m\n", steadyStats.getAverage());
        System.out.printf(" - Estado de Crecida   (200 m³/s): %.4f m\n", floodStats.getAverage());

        assertTrue(floodStats.getAverage() > steadyStats.getAverage(), "La profundidad media durante la crecida debe ser mayor.");
    }

    private void describeArray(String name, double[] data) {
        if (data == null || data.length == 0) {
            System.out.printf("\n--- %s ---\nDatos no disponibles o array vacío.\n", name);
            return;
        }
        System.out.printf("\n--- %s ---\n", name);
        DoubleSummaryStatistics stats = Arrays.stream(data).summaryStatistics();
        double mean = stats.getAverage();
        double variance = Arrays.stream(data).map(x -> (x - mean) * (x - mean)).average().orElse(0.0);
        double stdDev = Math.sqrt(variance);

        System.out.printf("Count:    %,d\n", stats.getCount());
        System.out.printf("Mean:     %.4f\n", mean);
        System.out.printf("Std Dev:  %.4f\n", stdDev);
        System.out.printf("Min:      %.4f\n", stats.getMin());
        System.out.printf("Max:      %.4f\n", stats.getMax());
        int headCount = Math.min(5, data.length);
        double[] head = Arrays.copyOfRange(data, 0, headCount);
        System.out.printf("Primeros %d valores: %s\n", headCount, Arrays.toString(head));
    }
}