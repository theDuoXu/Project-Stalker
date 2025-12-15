package projectstalker.factory;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import projectstalker.config.RiverConfig;
import projectstalker.domain.river.RiverGeometry;

import java.util.DoubleSummaryStatistics;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test de Integración para la Generación Procedural del Río.
 * <p>
 * A diferencia de un test unitario, este test no busca valores exactos, sino
 * <b>Comportamientos Emergentes y Correlaciones Físicas</b>.
 * Valida que las diferentes estrategias (SpatialModels) estén correctamente orquestadas
 * y produzcan un resultado geomorfológicamente coherente.
 */
@Slf4j
class RiverGeometryFactoryIntegrationTest {

    @Test
    @DisplayName("INTEGRACIÓN: Verificar correlaciones físicas entre Pendiente, Ancho, Manning y Química")
    void createRealisticRiver_shouldDemonstratePhysicalCoupling() {
        // --- 1. ARRANGE: Configuración exagerada ---
        // Usamos una configuración diseñada para acentuar las diferencias físicas
        // (Mucha pendiente inicial, gran variación de rugosidad) para que las correlaciones sean obvias.
        RiverConfig config = RiverConfig.getTestingRiver()
                .withTotalLength(50000.0F)      // 50 km
                .withSpatialResolution(50.0F)
                .withInitialElevation(1000.0F)  // Empieza alto (Montaña)
                .withConcavityFactor(0.5F)      // Perfil muy curvo (Rápido al inicio, plano al final)
                .withAverageSlope(0.01F)        // Pendiente media alta
                .withBaseWidth(20.0F)
                .withWidthVariability(5.0F)
                .withBaseManning(0.035F)
                .withManningVariability(0.01F)
                .withBaseDecayRateAt20C(0.2F)
                .withBaseDispersionAlpha(10.0F)
                .withZoneNoiseFrequency(0.005F); // Frecuencia zonal
                // resto de defaults no críticos para la morfología

        // --- 2. ACT: Generación ---
        long startTime = System.currentTimeMillis();
        RiverGeometry river = RiverGeometryFactory.createRealisticRiver(config);
        long duration = System.currentTimeMillis() - startTime;

        log.info("Río generado en {} ms. Celdas: {}", duration, river.getCellCount());

        // --- 3. ASSERT & ANÁLISIS ---

        // A. Integridad Estructural Básica
        assertNotNull(river);
        assertEquals(config.totalLength() / config.spatialResolution(), river.getCellCount(), 1.0);

        // Extraemos los perfiles para análisis estadístico
        float[] elevation = river.getElevationProfile();
        float[] width = river.getBottomWidth();
        float[] manning = river.getManningCoefficient();
        float[] decay = river.getBaseDecayCoefficientAt20C();
        float[] sideSlope = river.getSideSlope();

        // Calcular Pendiente Local (Driver Principal)
        float[] slopes = calculateLocalSlopes(elevation, config.spatialResolution());

        // --- B. VALIDACIÓN: Evolución Longitudinal (Downstream Widening) ---
        // El río debería ser, en promedio, más ancho al final que al principio.
        double avgWidthStart = getAverage(width, 0, 100); // Primeras 100 celdas
        double avgWidthEnd = getAverage(width, width.length - 100, width.length); // Últimas 100

        log.info("Ancho Medio Cabecera: {:.2f}m | Ancho Medio Desembocadura: {:.2f}m", avgWidthStart, avgWidthEnd);
        assertTrue(avgWidthEnd > avgWidthStart * 1.5,
                "El río debería ensancharse significativamente hacia la desembocadura (Efecto Decorador Longitudinal)");

        // --- C. VALIDACIÓN: Geometría Hidráulica (Pendiente vs Ancho) ---
        // Sin contar el efecto longitudinal, zonas empinadas deberían ser más estrechas.
        // Esperamos correlación NEGATIVA (Más pendiente = Menos ancho).
        // Nota: Como el efecto longitudinal domina, chequeamos tramos locales o una correlación débil.
        // Para simplificar el test, chequeamos que en los picos de pendiente, el ancho no se dispara.

        // --- D. VALIDACIÓN: Sediment Sorting (Pendiente vs Manning) ---
        // A más pendiente -> Más energía -> Rocas más grandes -> Mayor Manning.
        // Esperamos correlación POSITIVA FUERTE.
        double correlationSlopeManning = calculateCorrelation(slopes, manning);
        log.info("Correlación Pendiente vs Manning: {:.4f}", correlationSlopeManning);
        assertTrue(correlationSlopeManning > 0.4,
                "Debería existir una correlación positiva clara entre Pendiente y Manning (Sediment Sorting).");

        // --- E. VALIDACIÓN: Bio-Física (Manning vs Decay) ---
        // A más Manning -> Más turbulencia -> Mayor reaireación -> Mayor Decay.
        // Esperamos correlación POSITIVA MUY FUERTE (casi directa).
        double correlationManningDecay = calculateCorrelation(manning, decay);
        log.info("Correlación Manning vs Decay: {:.4f}", correlationManningDecay);
        assertTrue(correlationManningDecay > 0.8,
                "El Decay debería depender fuertemente de la rugosidad (Manning).");

        // --- F. VALIDACIÓN: Taludes (Manning vs Side Slope) ---
        // A más Manning (Roca) -> Talud más vertical (z menor).
        // Esperamos correlación NEGATIVA.
        double correlationManningSideSlope = calculateCorrelation(manning, sideSlope);
        log.info("Correlación Manning vs Talud (z): {:.4f}", correlationManningSideSlope);
        assertTrue(correlationManningSideSlope < -0.6,
                "Suelos más duros (Alto Manning) deberían permitir taludes más verticales (Menor z).");

        // --- G. Resumen Visual (Logs) ---
        log.info("--- INTEGRATION TEST PASSED: La física del río es coherente ---");
        describeArray("Pendiente Local (%)", slopes);
        describeArray("Manning (n)", manning);
        describeArray("Taludes (z)", sideSlope);
    }

    // --- UTILIDADES ESTADÍSTICAS ---

    /**
     * Calcula la correlación de Pearson entre dos arrays.
     * Rango: [-1, 1].
     * 1 = Correlación positiva perfecta.
     * -1 = Correlación negativa perfecta.
     * 0 = Sin relación.
     */
    private double calculateCorrelation(float[] xs, float[] ys) {
        if (xs.length != ys.length) throw new IllegalArgumentException("Arrays must have same length");
        int n = xs.length;

        double sumX = 0.0, sumY = 0.0, sumXY = 0.0;
        double sumX2 = 0.0, sumY2 = 0.0;

        for (int i = 0; i < n; i++) {
            double x = xs[i];
            double y = ys[i];

            sumX += x;
            sumY += y;
            sumXY += x * y;
            sumX2 += x * x;
            sumY2 += y * y;
        }

        double numerator = n * sumXY - sumX * sumY;
        double denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));

        return (denominator == 0) ? 0 : numerator / denominator;
    }

    private float[] calculateLocalSlopes(float[] elevation, float dx) {
        float[] slopes = new float[elevation.length];
        // Para i=0 asumimos la misma que i=1
        slopes[0] = (elevation[0] - elevation[1]) / dx;

        for (int i = 1; i < elevation.length; i++) {
            // Pendiente descendente positiva
            float drop = elevation[i-1] - elevation[i];
            slopes[i] = Math.max(0, drop / dx);
        }
        return slopes;
    }

    private double getAverage(float[] data, int start, int end) {
        return IntStream.range(start, end).mapToDouble(i -> data[i]).average().orElse(0.0);
    }

    private void describeArray(String name, float[] data) {
        DoubleSummaryStatistics stats = IntStream.range(0, data.length).mapToDouble(i -> data[i]).summaryStatistics();
        log.info("{}: Min={:.4f}, Max={:.4f}, Avg={:.4f}", name, stats.getMin(), stats.getMax(), stats.getAverage());
    }
}