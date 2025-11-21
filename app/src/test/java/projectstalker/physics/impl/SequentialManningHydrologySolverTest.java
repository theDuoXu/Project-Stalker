package projectstalker.physics.impl;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import projectstalker.config.RiverConfig;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;
import projectstalker.factory.RiverGeometryFactory;

import java.util.Arrays;
import java.util.DoubleSummaryStatistics;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Pruebas unitarias para la clase {@link SequentialManningHydrologySolver}.
 * Verifica que el solver calcula correctamente el siguiente estado hidrológico del río.
 */
class SequentialManningHydrologySolverTest {

    // Instancia del logger para esta clase.
    private static final Logger log = LoggerFactory.getLogger(SequentialManningHydrologySolverTest.class);

    private RiverConfig config;
    private RiverGeometry riverGeometry;
    private SequentialManningHydrologySolver solver;

    @BeforeEach
    void setUp() {
        // Usamos la configuración realista para un río tipo Tajo.
        config = RiverConfig.getTestingRiver();
        RiverGeometryFactory factory = new RiverGeometryFactory();
        riverGeometry = factory.createRealisticRiver(config);
        solver = new SequentialManningHydrologySolver();
        log.debug("Entorno de prueba para ManningHydrologySolver inicializado.");
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
        log.info("Iniciando test 'calculateNextState_fromDryBed' con caudal de {} m³/s sobre cauce seco.", inputDischarge);

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

        double baseTemp = config.averageAnnualTemperature();
        double tempAtHeadwater = nextState.getTemperatureAt(0);
        assertTrue(tempAtHeadwater < baseTemp, "La temperatura en la cabecera debería ser más fría que la base.");
        assertEquals(baseTemp - config.maxHeadwaterCoolingEffect(), tempAtHeadwater, 1.0, "La temperatura en la cabecera no refleja el efecto de enfriamiento esperado.");

        assertEquals(riverGeometry.getPhAt(50), nextState.getPhAt(50), 1e-6);
        assertEquals(0.0, nextState.getWaterDepthAt(cellCount - 1), "La última celda aún debería estar seca.");

        log.info("Test 'calculateNextState_fromDryBed' superado. El solver funciona de manera coherente.");

        // --- 4. Describe ---
        log.info("--- Resumen Estadístico del Primer Estado del Río (t=1) ---");
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
        log.info("--- FASE 1: Llenando el río con {} m³/s hasta el estado estacionario... ---", steadyDischarge);
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
        log.info("Río estabilizado después de {} iteraciones.", iterations);
        RiverState steadyState = currentState;

        // --- Análisis del Estado Estacionario ---
        double steadyOutputDischarge = riverGeometry.getCrossSectionalArea(lastCellIndex, steadyState.getWaterDepthAt(lastCellIndex)) * steadyState.getVelocityAt(lastCellIndex);
        assertEquals(steadyDischarge, steadyOutputDischarge, 5.0, "El caudal de salida en estado estacionario debería ser cercano a 150 m³/s.");
        describeArray("Profundidad (Estado Estacionario)", steadyState.waterDepth());

        // --- FASE 2: Propagación de una crecida con 200 m³/s ---
        log.info("--- FASE 2: Introduciendo una crecida de {} m³/s... ---", floodDischarge);
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
        log.info("Crecida estabilizada después de {} iteraciones adicionales.", iterations);
        RiverState floodState = currentState;

        // --- Análisis del Estado de la Crecida y Comparación ---
        assertEquals(floodDischarge, currentOutputDischarge, 1.0, "El caudal de salida final debería ser cercano a 200 m³/s.");
        describeArray("Profundidad (Estado de Crecida)", floodState.waterDepth());

        DoubleSummaryStatistics steadyStats = Arrays.stream(steadyState.waterDepth()).summaryStatistics();
        DoubleSummaryStatistics floodStats = Arrays.stream(floodState.waterDepth()).summaryStatistics();

        log.info("Comparación de Profundidad Media:");
        log.info(" - Estado Estacionario ({} m³/s): {} m", steadyDischarge, String.format("%.4f", steadyStats.getAverage()));
        log.info(" - Estado de Crecida   ({} m³/s): {} m", floodDischarge, String.format("%.4f", floodStats.getAverage()));

        assertTrue(floodStats.getAverage() > steadyStats.getAverage(), "La profundidad media durante la crecida debe ser mayor.");
    }

    private void describeArray(String name, double[] data) {
        if (data == null || data.length == 0) {
            log.warn("--- {} ---: Datos no disponibles o array vacío.", name);
            return;
        }
        log.info("--- {} ---", name);
        DoubleSummaryStatistics stats = Arrays.stream(data).summaryStatistics();
        double mean = stats.getAverage();
        double variance = Arrays.stream(data).map(x -> (x - mean) * (x - mean)).average().orElse(0.0);
        double stdDev = Math.sqrt(variance);

        log.info("  Count:    {}", stats.getCount());
        log.info("  Mean:     {}", String.format("%.4f", mean));
        log.info("  Std Dev:  {}", String.format("%.4f", stdDev));
        log.info("  Min:      {}", String.format("%.4f", stats.getMin()));
        log.info("  Max:      {}", String.format("%.4f", stats.getMax()));
        int headCount = Math.min(5, data.length);
        double[] head = Arrays.copyOfRange(data, 0, headCount);
        log.info("  Primeros {} valores: {}", headCount, Arrays.toString(head));
    }
}