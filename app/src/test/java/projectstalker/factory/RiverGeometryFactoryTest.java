package projectstalker.factory;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import projectstalker.config.RiverConfig;
import projectstalker.domain.river.RiverGeometry;

import java.util.Arrays;
import java.util.DoubleSummaryStatistics;

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
        RiverConfig config = new RiverConfig(
                12345L, 0.0f, 0.05f, 0.001f, 100000.0, 50.0, 200.0, 0.4, 0.0002,
                0.0001, 150.0, 40.0, 4.0, 1.5, 0.030, 0.005, 0.1, 0.05,
                15.0, 2.0, 8.0, 14.0, 7.5, 0.5,
                4.0, 20000.0, 1.5, 1.0, 0.25
        );
        RiverGeometryFactory factory = new RiverGeometryFactory();
        int expectedCellCount = (int) (config.totalLength() / config.spatialResolution());
        log.debug("Configuración de río creada para el test. Se esperan {} celdas.", expectedCellCount);


        // --- 2. Act (Actuar) ---
        RiverGeometry river = factory.createRealisticRiver(config);
        log.debug("RiverGeometry creada satisfactoriamente por la fábrica.");


        // --- 3. Assert (Verificar) ---
        assertNotNull(river, "El objeto RiverGeometry no debería ser nulo.");
        assertEquals(expectedCellCount, river.getCellCount(), "El número de celdas no es el esperado.");
        assertEquals(config.spatialResolution(), river.getDx(), "La resolución espacial (dx) no coincide.");
        assertEquals(config.initialElevation(), river.cloneElevationProfile()[0], 1e-6, "La elevación inicial no coincide.");

        log.info("Test de aserciones superado. El objeto RiverGeometry es estructuralmente válido.");
        // El método toString() de RiverGeometry está bien diseñado, por lo que podemos registrarlo directamente.
        log.info("Resumen del objeto RiverGeometry:\n{}", river);


        // --- 4. Describe (Análisis Estadístico) ---
        log.info("--- Resumen Estadístico de la Geometría del Río ---");

        // Obtenemos los arrays una sola vez para optimizar.
        double[] elevationProfile = river.cloneElevationProfile();
        double[] bottomWidth = river.cloneBottomWidth();
        double[] sideSlope = river.cloneSideSlope();
        double[] manningCoefficient = river.cloneManningCoefficient();
        double[] decayCoefficient = river.cloneBaseDecayCoefficientAt20C();
        double[] phProfile = river.clonePhProfile();

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
    private void describeArray(String name, double[] data) {
        if (data == null || data.length == 0) {
            log.warn("--- {} ---: Datos no disponibles o array vacío.", name);
            return;
        }

        log.info("--- {} ---", name);
        DoubleSummaryStatistics stats = Arrays.stream(data).summaryStatistics();
        double mean = stats.getAverage();
        // El cálculo de la varianza se hace una vez para ser eficiente.
        double variance = Arrays.stream(data)
                .map(x -> (x - mean) * (x - mean))
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
        double[] head = Arrays.copyOfRange(data, 0, headCount);
        log.info("  Primeros {} valores: {}", headCount, Arrays.toString(head));
    }
}