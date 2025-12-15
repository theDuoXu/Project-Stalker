package projectstalker.physics.model;

import lombok.extern.slf4j.Slf4j;
import projectstalker.config.RiverConfig;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.utils.FastNoiseLite;

/**
 * Modela la temperatura del agua a lo largo de un río.
 * <p>
 * Calcula el perfil de temperaturas basándose en una combinación de factores:
 * <ul>
 * <li><b>Ciclos temporales:</b> Variaciones estacionales y diarias.</li>
 * <li><b>Gradiente longitudinal:</b> Enfriamiento del agua cerca de la cabecera.</li>
 * <li><b>Efectos geomorfológicos:</b> Influencia del ancho del cauce y la pendiente.</li>
 * <li><b>Variabilidad local:</b> Ruido Perlin para simular fluctuaciones naturales.</li>
 * </ul>
 */
@Slf4j
public class RiverTemperatureModel implements TemperatureModel{

    private static final double SECONDS_IN_A_DAY = 24.0 * 3600.0;
    private static final double DAYS_IN_A_YEAR = 365.25;

    private final RiverConfig config;
    private final RiverGeometry geometry;
    private final FastNoiseLite tempNoise;

    /**
     * Constructor para el modelo de temperatura del río.
     *
     * @param config   La configuración global de la simulación.
     * @param geometry La geometría inmutable del cauce del río.
     */
    public RiverTemperatureModel(RiverConfig config, RiverGeometry geometry) {
        this.config = config;
        this.geometry = geometry;
        this.tempNoise = new FastNoiseLite((int) config.seed() + 2); // Semilla específica para temperatura
        this.tempNoise.SetNoiseType(FastNoiseLite.NoiseType.Perlin);
        this.tempNoise.SetFrequency(0.2f);
    }

    /**
     * Calcula el perfil de temperaturas completo para un instante de tiempo dado.
     *
     * @param currentTimeInSeconds El tiempo absoluto de la simulación.
     * @return Un array de doubles con la temperatura para cada celda del río.
     */
    @Override
    public float[] generateProfile(double currentTimeInSeconds) {
        final int cellCount = geometry.getCellCount();
        float[] temperatures = new float[cellCount];

        // 1. Se calcula la temperatura base (dependiente del tiempo) una sola vez.
        final float baseTemp = calculateBaseTemperature(currentTimeInSeconds);

        // 2. Se calcula la temperatura final para cada celda sumando los diferentes efectos.
        for (int i = 0; i < cellCount; i++) {
            float headwaterEffect = calculateHeadwaterCooling(i);
            float geomorphologyEffect = calculateGeomorphologyEffect(i);
            float noiseEffect = calculateNoiseEffect(i);

            temperatures[i] = baseTemp + headwaterEffect + geomorphologyEffect + noiseEffect;
        }

        return temperatures;
    }

    /**
     * Calcula la temperatura base del agua influenciada por los ciclos estacionales y diarios.
     */
    private float calculateBaseTemperature(double currentTimeInSeconds) {
        final double dayOfYear = (currentTimeInSeconds / SECONDS_IN_A_DAY) % DAYS_IN_A_YEAR;
        final double seasonalCycle = Math.sin((dayOfYear / DAYS_IN_A_YEAR) * 2.0 * Math.PI);
        final double baseSeasonalTemp = config.averageAnnualTemperature() + config.seasonalTempVariation() * seasonalCycle;

        final double secondOfDay = currentTimeInSeconds % SECONDS_IN_A_DAY;
        final double dailyCycle = Math.sin((secondOfDay / SECONDS_IN_A_DAY) * 2.0 * Math.PI);

        return (float) (baseSeasonalTemp + config.dailyTempVariation() * dailyCycle);
    }

    /**
     * Calcula el efecto de enfriamiento en la cabecera del río.
     */
    private float calculateHeadwaterCooling(int cellIndex) {
        double position = cellIndex * geometry.getSpatialResolution();
        double gradientFactor = Math.max(0, 1.0 - (position / config.headwaterCoolingDistance()));
        return (float) (-config.maxHeadwaterCoolingEffect() * gradientFactor);
    }

    /**
     * Calcula el efecto combinado del ancho del cauce y la pendiente sobre la temperatura.
     */
    private float calculateGeomorphologyEffect(int cellIndex) {
        // Efecto del ancho: ríos más anchos se calientan más.
        double relativeWidth = geometry.getWidthAt(cellIndex) / config.baseWidth();
        double widthEffect = config.widthHeatingFactor() * Math.max(0, relativeWidth - 1.0);

        // Efecto de la pendiente: pendientes mayores aumentan la turbulencia y el enfriamiento.
        double relativeSlope = geometry.getBedSlopeAt(cellIndex) / config.averageSlope();
        double slopeEffect = -config.slopeCoolingFactor() * Math.max(0, relativeSlope - 1.0);

        return (float) (widthEffect + slopeEffect);
    }

    /**
     * Calcula una variación local aleatoria usando ruido Perlin.
     */
    private float calculateNoiseEffect(int cellIndex) {
        return tempNoise.GetNoise(cellIndex, 0) * config.temperatureNoiseAmplitude();
    }
//    /**
//     * Genera y muestra un gráfico interactivo del perfil de temperaturas a lo largo del río
//     * para un instante de tiempo específico.
//     * <p>
//     * El método bloquea el hilo que lo llama hasta que el usuario cierra la ventana.
//     *
//     * @param currentTimeInSeconds El instante de tiempo para el cual se calculará y mostrará el perfil.
//     * @throws InterruptedException si el hilo es interrumpido mientras espera.
//     */
//    public void displayProfileChart(double currentTimeInSeconds) throws InterruptedException {
//        // --- 1. Generación de Datos ---
//        float[] fTemperatureProfile = this.calculate(currentTimeInSeconds);
//        double[] temperatureProfile = IntStream.range(0, fTemperatureProfile.length)
//                .mapToDouble(i -> fTemperatureProfile[i])
//                .toArray(); // Y-Axis cast to double
//        double[] distanceInMeters = DoubleStream.iterate(0, d -> d + geometry.getSpatialResolution())
//                .limit(temperatureProfile.length)
//                .toArray(); // X-Axis
//
//        // --- 2. Creación del Gráfico con XChart ---
//        XYChart chart = new XYChartBuilder()
//                .width(900)
//                .height(600)
//                .title("Perfil de Temperatura del Río")
//                .xAxisTitle("Distancia desde la cabecera (m)")
//                .yAxisTitle("Temperatura (°C)")
//                .build();
//
//        chart.getStyler().setMarkerSize(0);
//        chart.addSeries("Temperatura", distanceInMeters, temperatureProfile);
//
//        // --- 3. Visualización Sincronizada ---
//        final CountDownLatch latch = new CountDownLatch(1);
//        JFrame frame = new SwingWrapper<>(chart).displayChart();
//
//        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
//        frame.addWindowListener(new WindowAdapter() {
//            @Override
//            public void windowClosed(WindowEvent e) {
//                log.info("Ventana del gráfico de temperatura cerrada. Se libera el bloqueo.");
//                latch.countDown();
//            }
//        });
//
//        log.info("Mostrando gráfico de temperatura. El hilo esperará al cierre de la ventana.");
//        latch.await();
//    }
}