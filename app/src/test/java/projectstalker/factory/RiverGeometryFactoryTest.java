package projectstalker.factory;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import projectstalker.config.RiverConfig;
import projectstalker.domain.river.RiverGeometry;

import java.util.Arrays;
import java.util.DoubleSummaryStatistics;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Pruebas unitarias para la clase {@link RiverGeometryFactory}.
 */
class RiverGeometryFactoryTest {

    @Test
    @DisplayName("Debería crear una geometría de río consistente y mostrar su resumen estadístico")
    void createRealisticRiver_shouldReturnConsistentGeometry() {
        // --- 1. Arrange (Preparar) ---
        // Configuración para un río grande y maduro, tipo Tajo, con los nuevos parámetros de temperatura.
        RiverConfig config = new RiverConfig(
                12345L,      // seed
                0.0f,        // noiseFrequency
                0.05f,       // detailNoiseFrequency
                0.001f,      // zoneNoiseFrequency
                100000.0,    // totalLength
                50.0,        // spatialResolution
                200.0,       // initialElevation
                0.4,         // concavityFactor
                0.0002,      // averageSlope (corregido a un valor más suave para un río de llanura)
                0.0001,      // slopeVariability
                150.0,       // baseWidth
                40.0,        // widthVariability
                4.0,         // baseSideSlope
                1.5,         // sideSlopeVariability
                0.030,       // baseManning
                0.005,       // manningVariability
                0.1,         // baseDecayRateAt20C
                0.05,        // decayRateVariability
                // --- Parámetros de Calidad de Agua (Temporales) ---
                15.0,        // baseTemperature
                2.0,         // dailyTempVariation
                8.0,         // seasonalTempVariation
                14.0,        // averageAnnualTemperature
                7.5,         // basePh
                0.5,         // phVariability
                // --- Parámetros de Modelo de Temperatura Espacial (AÑADIDOS) ---
                4.0,         // maxHeadwaterCoolingEffect: 4°C más frío en el nacimiento.
                20000.0,     // headwaterCoolingDistance: El efecto se disipa en 20km.
                1.5,         // widthHeatingFactor: Tramos anchos se calientan hasta 1.5°C extra.
                1.0,         // slopeCoolingFactor: Tramos rápidos se enfrían hasta 1.0°C extra.
                0.25         // temperatureNoiseAmplitude: +/- 0.25°C de variación aleatoria.
        );
        RiverGeometryFactory factory = new RiverGeometryFactory();
        int expectedCellCount = (int) (config.totalLength() / config.spatialResolution());

        // --- 2. Act (Actuar) ---
        RiverGeometry river = factory.createRealisticRiver(config);

        // --- 3. Assert (Verificar) ---
        assertNotNull(river, "El objeto RiverGeometry no debería ser nulo.");
        assertEquals(expectedCellCount, river.getCellCount(), "El número de celdas no es el esperado.");
        assertEquals(config.spatialResolution(), river.getDx(), "La resolución espacial (dx) no coincide.");
        assertEquals(config.initialElevation(), river.cloneElevationProfile()[0], 1e-6, "La elevación inicial no coincide.");

        System.out.println("Test de aserciones superado con éxito. El objeto RiverGeometry es válido.");
        System.out.println(river); // Imprime el resumen general del objeto

        // --- 4. Describe (Análisis Estadístico) ---
        System.out.println("\n--- Resumen Estadístico de la Geometría del Río (estilo pandas.describe) ---");

        // Obtenemos los arrays una sola vez para optimizar, como sugeriste.
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
     * Calcula e imprime un resumen estadístico para un array de doubles.
     */
    private void describeArray(String name, double[] data) {
        if (data == null || data.length == 0) {
            System.out.printf("\n--- %s ---\nDatos no disponibles o array vacío.\n", name);
            return;
        }

        System.out.printf("\n--- %s ---\n", name);
        DoubleSummaryStatistics stats = Arrays.stream(data).summaryStatistics();
        double mean = stats.getAverage();
        double variance = Arrays.stream(data)
                .map(x -> (x - mean) * (x - mean))
                .average()
                .orElse(0.0);
        double stdDev = Math.sqrt(variance);

        System.out.printf("Count:    %,d\n", stats.getCount());
        System.out.printf("Mean:     %.4f\n", mean);
        System.out.printf("Std Dev:  %.4f\n", stdDev);
        System.out.printf("Variance: %.4f\n", variance);
        System.out.printf("Min:      %.4f\n", stats.getMin());
        System.out.printf("Max:      %.4f\n", stats.getMax());
        int headCount = Math.min(5, data.length);
        double[] head = Arrays.copyOfRange(data, 0, headCount);
        System.out.printf("Primeros %d valores: %s\n", headCount, Arrays.toString(head));
    }
}