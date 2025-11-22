package projectstalker.factory;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import projectstalker.config.RiverConfig;
import projectstalker.domain.river.RiverGeometry;

import java.util.Arrays;
import java.util.DoubleSummaryStatistics;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Pruebas unitarias para la clase {@link RiverGeometryFactory}.
 */
class RiverGeometryFactoryTest {

    // Instancia del logger para esta clase de prueba.
    private static final Logger log = LoggerFactory.getLogger(RiverGeometryFactoryTest.class);

    @Test
    @DisplayName("Debería crear una geometría de río consistente y mostrar su resumen estadístico")
    void createRealisticRiver_shouldReturnConsistentGeometry() {
        // --- 1. Arrange (Preparar) ---
        // Configuración para un río grande y maduro, tipo Tajo, con los nuevos parámetros de temperatura.

        RiverConfig config =
                RiverConfig.builder()
                        .seed(12345L)
                        .noiseFrequency(0.0f)
                        .detailNoiseFrequency(0.05f)
                        .zoneNoiseFrequency(0.001f)
                        .totalLength(100000.0F)
                        .spatialResolution(50.0F)
                        .initialElevation(200)
                        .concavityFactor(0.4F)
                        .averageSlope(0.0002F)
                        .slopeVariability(0.0001F)
                        .baseWidth(150.0F)
                        .widthVariability(40.0F)
                        .baseSideSlope(4.0F)
                        .sideSlopeVariability(1.5F)
                        .baseManning(0.030F)
                        .manningVariability(0.005F)
                        .baseDecayRateAt20C(0.1F)
                        .decayRateVariability(0.05F)
                        .baseDispersionAlpha(10)
                        .alphaVariability(2)
                        .baseTemperature(15)
                        .dailyTempVariation(2.0F)
                        .seasonalTempVariation(8.0F)
                        .averageAnnualTemperature(14.0F)
                        .basePh(7.5F)
                        .phVariability(0.5F)
                        .maxHeadwaterCoolingEffect(4.0F)
                        .headwaterCoolingDistance(20000.0F)
                        .widthHeatingFactor(1.5F)
                        .slopeCoolingFactor(1.0F)
                        .temperatureNoiseAmplitude(0.25F)
                        .build();

        RiverGeometryFactory factory = new RiverGeometryFactory();
        int expectedCellCount = (int) (config.totalLength() / config.spatialResolution());
        log.debug("Configuración de río creada para el test. Se esperan {} celdas.", expectedCellCount);


        // --- 2. Act (Actuar) ---
        RiverGeometry river = factory.createRealisticRiver(config);
        log.debug("RiverGeometry creada satisfactoriamente por la fábrica.");


        // --- 3. Assert (Verificar) ---
        assertNotNull(river, "El objeto RiverGeometry no debería ser nulo.");
        assertEquals(expectedCellCount, river.getCellCount(), "El número de celdas no es el esperado.");
        assertEquals(config.spatialResolution(), river.getSpatial_resolution(), "La resolución espacial (dx) no coincide.");
        assertEquals(config.initialElevation(), river.cloneElevationProfile()[0], 1e-6, "La elevación inicial no coincide.");

        log.info("Test de aserciones superado. El objeto RiverGeometry es estructuralmente válido.");
        // El método toString() de RiverGeometry está bien diseñado, por lo que podemos registrarlo directamente.
        log.info("Resumen del objeto RiverGeometry:\n{}", river);


        // --- 4. Describe (Análisis Estadístico) ---
        log.info("--- Resumen Estadístico de la Geometría del Río ---");

        // Obtenemos los arrays una sola vez para optimizar.
        float[] elevationProfile = river.cloneElevationProfile();
        float[] bottomWidth = river.cloneBottomWidth();
        float[] sideSlope = river.cloneSideSlope();
        float[] manningCoefficient = river.cloneManningCoefficient();
        float[] decayCoefficient = river.cloneBaseDecayCoefficientAt20C();
        float[] phProfile = river.clonePhProfile();

        describeArray("Perfil de Elevación (m)", elevationProfile);
        describeArray("Ancho del Fondo (m)", bottomWidth);
        describeArray("Pendiente de Taludes", sideSlope);
        describeArray("Coeficiente de Manning", manningCoefficient);
        describeArray("Tasa de Decaimiento a 20°C", decayCoefficient);
        describeArray("Perfil de pH", phProfile);
    }

    /**
     * Calcula y registra un resumen estadístico para un array de doubles.
     */
    private void describeArray(String name, float[] data) {
        if (data == null || data.length == 0) {
            log.warn("--- {} ---: Datos no disponibles o array vacío.", name);
            return;
        }

        log.info("--- {} ---", name);
        DoubleSummaryStatistics stats = IntStream.range(0, data.length).mapToDouble(i -> data[i]).summaryStatistics();
        double mean = stats.getAverage();
        // El cálculo de la varianza se hace una vez para ser eficiente.
        double variance = IntStream.range(0, data.length)
                .mapToDouble(x -> (data[x] - mean) * (data[x] - mean))
                .average()
                .orElse(0.0);
        double stdDev = Math.sqrt(variance);

        // Usamos String.format para mantener la precisión de los decimales en los logs.
        log.info("  Count:    {}", stats.getCount());
        log.info("  Mean:     {}", String.format("%.4f", mean));
        log.info("  Std Dev:  {}", String.format("%.4f", stdDev));
        log.info("  Variance: {}", String.format("%.4f", variance));
        log.info("  Min:      {}", String.format("%.4f", stats.getMin()));
        log.info("  Max:      {}", String.format("%.4f", stats.getMax()));
        int headCount = Math.min(5, data.length);
        float[] head = Arrays.copyOfRange(data, 0, headCount);
        log.info("  Primeros {} valores: {}", headCount, Arrays.toString(head));
    }
}