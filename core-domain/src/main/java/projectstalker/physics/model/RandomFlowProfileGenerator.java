package projectstalker.physics.model;

import lombok.RequiredArgsConstructor;
import lombok.With;
import lombok.extern.slf4j.Slf4j;
import projectstalker.config.SimulationConfig;
import projectstalker.utils.FastNoiseLite;

/**
 * Genera un perfil de caudal (hidrograma) variable a lo largo del tiempo utilizando ruido Perlin.
 * Esto permite simular hidrogramas realistas con variaciones suaves y creíbles,
 * sirviendo como condición de contorno de entrada para simulaciones hidrológicas.
 */
@Slf4j
@With
@RequiredArgsConstructor
public class RandomFlowProfileGenerator implements TimeSeriesGenerator {

    /**
     * El caudal promedio (en m³/s) sobre el cual se aplican las variaciones.
     */
    private final double baseDischarge;

    /**
     * La magnitud máxima de la variación (en m³/s) sobre el caudal base.
     */
    private final double noiseAmplitude;

    /**
     * Instancia del generador de ruido para crear las variaciones de caudal.
     */
    private final FastNoiseLite noise;

    /**
     * Constructor para crear un generador de caudal con una configuración específica.
     *
     * @param seed Semilla para la generación de ruido, permite reproducibilidad.
     * @param baseDischarge Caudal promedio o base en m³/s.
     * @param noiseAmplitude Magnitud de la variación en m³/s.
     * @param noiseFrequency Frecuencia de la variación (valores bajos para cambios lentos).
     */
    public RandomFlowProfileGenerator(int seed, double baseDischarge, double noiseAmplitude, float noiseFrequency) {
        this.baseDischarge = baseDischarge;
        this.noiseAmplitude = noiseAmplitude;
        this.noise = new FastNoiseLite(seed);
        this.noise.SetNoiseType(FastNoiseLite.NoiseType.Perlin);
        this.noise.SetFrequency(noiseFrequency);
    }

    /**
     * Constructor para el generador de caudal.
     *
     * @param seed       Semilla para la generación de ruido. Determina la secuencia de valores.
     * @param flowConfig Contenedor de configuraciones
     */
    public RandomFlowProfileGenerator(int seed, SimulationConfig.FlowConfig flowConfig) {
        this.noise = new FastNoiseLite(seed);
        this.noise.SetNoiseType(FastNoiseLite.NoiseType.Perlin);
        this.noise.SetFrequency(flowConfig.getNoiseFrequency());

        this.baseDischarge = flowConfig.getBaseDischarge();
        this.noiseAmplitude = flowConfig.getNoiseAmplitude();
    }


    /**
     * Calcula el valor del caudal en un instante de tiempo específico.
     *
     * @param timeInSeconds El punto en el tiempo (en segundos) para el cual se calcula el caudal.
     * @return El valor del caudal (en m³/s), garantizado como no negativo.
     */
    public float getDischargeAt(double timeInSeconds) {
        // Obtiene un valor de ruido Perlin 1D en el rango [-1, 1].
        float noiseValue = noise.GetNoise((float) timeInSeconds, 0.0f);

        // Calcula la variación y la suma al caudal base.
        double discharge = baseDischarge + noiseValue * noiseAmplitude;

        // Asegura que el caudal nunca sea negativo.
        return (float) Math.max(0, discharge);
    }

    /**
     * Genera una serie temporal de valores de caudal para un intervalo de tiempo dado.
     *
     * @param startTimeInSeconds El tiempo de inicio del perfil (en segundos).
     * @param endTimeInSeconds   El tiempo final del perfil (en segundos).
     * @param timeStepInSeconds  El incremento de tiempo entre cada punto del perfil (en segundos).
     * @return Un array de doubles que representa el caudal en cada paso de tiempo.
     * @throws IllegalArgumentException si el tiempo de fin es anterior al de inicio o si el paso de tiempo no es positivo.
     */
    @Override
    public float[] generateProfile(double startTimeInSeconds, double endTimeInSeconds, double timeStepInSeconds) {
        if (endTimeInSeconds < startTimeInSeconds) {
            throw new IllegalArgumentException("El tiempo final no puede ser menor que el tiempo inicial.");
        }
        if (timeStepInSeconds <= 0) {
            throw new IllegalArgumentException("El paso de tiempo debe ser un valor positivo.");
        }

        // Se calcula el número de pasos necesarios. +1 para incluir el punto final.
        int numSteps = (int) Math.floor((endTimeInSeconds - startTimeInSeconds) / timeStepInSeconds) + 1;
        float[] dischargeProfile = new float[numSteps];

        double currentTime = startTimeInSeconds;
        for (int i = 0; i < numSteps; i++) {
            dischargeProfile[i] = getDischargeAt(currentTime);
            currentTime += timeStepInSeconds;
        }

        return dischargeProfile;
    }
    /**
     * Genera una serie temporal de valores de caudal comenzando desde el tiempo cero.
     *
     * @param durationInSeconds  La duración total del perfil (en segundos).
     * @param timeStepInSeconds  El incremento de tiempo entre cada punto (en segundos).
     * @return Un array de doubles que representa el caudal en cada paso de tiempo.
     */
    @Override
    public float[] generateProfile(double durationInSeconds, double timeStepInSeconds) {
        return generateProfile(0, durationInSeconds, timeStepInSeconds);
    }

    /**
     * Calcula el volumen total de agua (en m³) que pasa en un intervalo de tiempo.
     * Se aproxima mediante la suma de los caudales en cada paso de tiempo multiplicada por el incremento.
     *
     * @param startTimeInSeconds El tiempo de inicio del intervalo (en segundos).
     * @param endTimeInSeconds   El tiempo final del intervalo (en segundos).
     * @param timeStepInSeconds  El incremento de tiempo para la integración numérica (en segundos).
     * @return El volumen total estimado en metros cúbicos.
     */
    public double getTotalVolume(double startTimeInSeconds, double endTimeInSeconds, double timeStepInSeconds) {
        float[] profile = generateProfile(startTimeInSeconds, endTimeInSeconds, timeStepInSeconds);
        double totalVolume = 0;
        for (double discharge : profile) {
            totalVolume += discharge * timeStepInSeconds;
        }
        return totalVolume;
    }
    /**
     * Encuentra el caudal pico (máximo) en un intervalo de tiempo.
     *
     * @param startTimeInSeconds El tiempo de inicio del intervalo (en segundos).
     * @param endTimeInSeconds   El tiempo final del intervalo (en segundos).
     * @param timeStepInSeconds  El paso de tiempo para muestrear el perfil (en segundos).
     * @return El valor del caudal máximo encontrado en m³/s.
     */
    public double getPeakDischarge(double startTimeInSeconds, double endTimeInSeconds, double timeStepInSeconds) {
        float[] profile = generateProfile(startTimeInSeconds, endTimeInSeconds, timeStepInSeconds);
        double maxDischarge = 0.0;
        for (double discharge : profile) {
            if (discharge > maxDischarge) {
                maxDischarge = discharge;
            }
        }
        return maxDischarge;
    }

//    /**
//     * Genera y muestra un gráfico interactivo del perfil de caudal para un intervalo de tiempo.
//     * <p>
//     * El method bloquea el hilo desde el que se llama hasta que el usuario cierra la ventana del gráfico.
//     * Esto es útil para visualizaciones rápidas durante pruebas o desde un method main.
//     *
//     * @param startTimeInSeconds El tiempo de inicio del perfil a visualizar (en segundos).
//     * @param endTimeInSeconds   El tiempo final del perfil a visualizar (en segundos).
//     * @param timeStepInSeconds  El incremento de tiempo entre cada punto del perfil (en segundos).
//     * @throws InterruptedException si el hilo es interrumpido mientras espera.
//     */
//    public void displayProfileChart(double startTimeInSeconds, double endTimeInSeconds, double timeStepInSeconds) throws InterruptedException {
//        // --- 1. Generación de Datos ---
//        double[] dischargeProfile = this.generateProfile(startTimeInSeconds, endTimeInSeconds, timeStepInSeconds);
//        // El eje X se muestra en horas para facilitar la lectura
//        double[] timeInHours = DoubleStream.iterate(startTimeInSeconds / 3600.0, t -> t + timeStepInSeconds / 3600.0)
//                .limit(dischargeProfile.length)
//                .toArray();
//
//        // --- 2. Creación del Gráfico con XChart ---
//        XYChart chart = new XYChartBuilder()
//                .width(800)
//                .height(600)
//                .title("Hidrograma Generado (Flow Profile)")
//                .xAxisTitle("Tiempo (horas)")
//                .yAxisTitle("Caudal (m³/s)")
//                .build();
//
//        chart.getStyler().setLegendPosition(Styler.LegendPosition.InsideNW);
//        chart.getStyler().setMarkerSize(0);
//        chart.getStyler().setYAxisMin(0.0);
//
//        chart.addSeries(
//                String.format("Caudal (Base: %.1f, Amplitud: %.1f)", this.baseDischarge, this.noiseAmplitude),
//                timeInHours,
//                dischargeProfile
//        );
//
//        // --- 3. Visualización Sincronizada con CountDownLatch ---
//        final CountDownLatch latch = new CountDownLatch(1);
//        JFrame frame = new SwingWrapper<>(chart).displayChart();
//
//        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE); // Importante para que se active windowClosed
//        frame.addWindowListener(new WindowAdapter() {
//            @Override
//            public void windowClosed(WindowEvent e) {
//                log.info("Ventana del gráfico cerrada. Se libera el bloqueo.");
//                latch.countDown();
//            }
//        });
//
//        log.info("Mostrando gráfico. El hilo actual esperará hasta que la ventana se cierre.");
//        latch.await();
//    }
}