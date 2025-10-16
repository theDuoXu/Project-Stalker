package projectstalker.physics.simulator;

import lombok.Builder;
import lombok.Getter;
import lombok.With;
import lombok.extern.slf4j.Slf4j;
import projectstalker.config.RiverConfig;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;
import projectstalker.factory.RiverGeometryFactory;
import projectstalker.physics.impl.ManningHydrologySolver;
import projectstalker.physics.solver.IHydrologySolver;
import projectstalker.utils.FastNoiseLite;

/**
 * Orquesta la simulación hidrológica del río, utilizando un modelo híbrido.
 * Comienza con un solver secuencial en Java (CPU) para la fase de llenado inicial
 * y transiciona a un solver masivamente paralelo en CUDA (GPU) una vez que el
 * río está completamente conectado hidrológicamente.
 */
@Slf4j
public class ManningSimulator {

    // --- Carga de la librería nativa (JNI) ---
    static {
        if ("true".equalsIgnoreCase(System.getProperty("projectstalker.native.enabled"))) {
            try {
                log.info("Native library loading ENABLED. Attempting to load 'manning_solver'...");
                System.loadLibrary("manning_solver");
                log.info("Native library 'manning_solver' loaded successfully.");
            } catch (UnsatisfiedLinkError e) {
                log.error("FATAL: Native library 'manning_solver' failed to load. " +
                        "Ensure the library is in the java.library.path.", e);
//                System.exit(1);
            }
        } else {
            log.warn("Native library loading is DISABLED. Running in pure Java/mock mode.");
        }
    }

    // --- Miembros de la Simulación ---
    /**
     * Configuración inmutable del río y la simulación.
     */
    private final RiverConfig config;
    /**
     * Geometría inmutable del cauce del río.
     */
    @Getter
    private final RiverGeometry geometry;
    /**
     * Implementación del solver hidrológico que se ejecuta en la CPU para la fase de llenado.
     */
    private final IHydrologySolver cpuSolver;
    /**
     * Generador de caudal de entrada variable para la simulación.
     */
    private final FlowProfileGenerator flowGenerator;
    /**
     * Generador de ruido para la variabilidad espacial de la temperatura.
     */
    private final FastNoiseLite tempNoise;

    // --- Estado de la Simulación ---
    /**
     * El estado actual del río (profundidad, velocidad, etc.), que se actualiza en cada paso.
     */
    @Getter
    private RiverState currentState;
    /**
     * El tiempo transcurrido en la simulación, en segundos.
     */
    @Getter
    private double currentTimeInSeconds;
    /**
     * Flag que indica si la simulación ha transicionado al modo acelerado por GPU.
     */
    @Getter
    private boolean isGpuAccelerated = false;

    // --- Métricas de Rendimiento ---
    /**
     * Tiempo total en nanosegundos consumido por la fase de llenado en CPU.
     */
    private long cpuFillTimeNanos = 0;
    /**
     * Número total de iteraciones realizadas durante la fase de llenado en CPU.
     */
    private int cpuFillIterations = 0;

    /**
     * Constructor del simulador.
     *
     * @param config La configuración global para el río y la simulación.
     */
    public ManningSimulator(RiverConfig config) {
        this.config = config;
        this.geometry = new RiverGeometryFactory().createRealisticRiver(config);
        this.cpuSolver = new ManningHydrologySolver();
        this.flowGenerator = new FlowProfileGenerator((int) config.seed(), 150.0, 50.0, 0.0001f);

        // Inicializa el generador de ruido para la temperatura con una semilla diferente.
        this.tempNoise = new FastNoiseLite((int) config.seed() + 2);
        this.tempNoise.SetNoiseType(FastNoiseLite.NoiseType.Perlin);
        this.tempNoise.SetFrequency(0.2f);

        int cellCount = geometry.getCellCount();
        // El río comienza completamente seco.
        this.currentState = new RiverState(new double[cellCount], new double[cellCount], new double[cellCount], new double[cellCount]);
        this.currentTimeInSeconds = 0.0;
        log.info("ManningSimulator inicializado. Esperando en modo CPU.");
    }

    /**
     * Avanza la simulación un paso de tiempo.
     * Delega el cálculo al motor de CPU o GPU según el estado actual.
     *
     * @param deltaTimeInSeconds El incremento de tiempo en segundos.
     */
    public void advanceTimeStep(double deltaTimeInSeconds) {
        if (!isGpuAccelerated) {
            runCpuStep(deltaTimeInSeconds);
        } else {
            runGpuStep(deltaTimeInSeconds);
        }
        this.currentTimeInSeconds += deltaTimeInSeconds;
    }

    /**
     * Ejecuta un paso de simulación usando el solver secuencial de Java (CPU).
     * Este modo se utiliza para llenar el río hasta que el agua llega al final,
     * momento en el que se transiciona al modo GPU.
     */
    private void runCpuStep(double deltaTimeInSeconds) {
        long startTime = System.nanoTime();
        double inputDischarge = flowGenerator.getDischargeAt(currentTimeInSeconds);

        this.currentState = cpuSolver.calculateNextState(currentState, geometry, config, currentTimeInSeconds, inputDischarge);
        this.cpuFillIterations++;
        this.cpuFillTimeNanos += (System.nanoTime() - startTime);

        // --- Comprobación de Transición a GPU ---
        int lastCellIndex = geometry.getCellCount() - 1;
        if (currentState.getWaterDepthAt(lastCellIndex) > 1e-6) {
            this.isGpuAccelerated = true;
            log.info("--- Transición a MODO GPU ---");
            log.info("El frente de agua ha alcanzado el final del río.");
            log.info("Tiempo de llenado en CPU: {} ms", cpuFillTimeNanos / 1_000_000);
            log.info("Iteraciones en CPU: {}", cpuFillIterations);
        }
    }

    /**
     * Orquesta un paso de simulación usando el solver paralelo de CUDA (GPU).
     * Prepara y sanitiza todos los datos antes de enviarlos a la GPU a través de JNI.
     */
    private void runGpuStep(double deltaTimeInSeconds) {
        final int cellCount = geometry.getCellCount();

        // --- 1. Pre-cómputo de Caudales ---
        float[] targetDischarges = precomputeTargetDischarges(cellCount);

        // --- 2. Sanitización de Datos para la GPU ---
        float[] initialDepthGuesses = sanitizeInitialDepths(cellCount);
        float[] bottomWidths = toFloatArray(geometry.cloneBottomWidth());
        float[] sideSlopes = toFloatArray(geometry.cloneSideSlope());
        float[] manningCoefficients = toFloatArray(geometry.cloneManningCoefficient());
        float[] bedSlopes = calculateAndSanitizeBedSlopes();

        // --- 3. Llamada al JNI ---
        float[] gpuResults = solveManningGpu(
                targetDischarges,
                initialDepthGuesses,
                bottomWidths,
                sideSlopes,
                manningCoefficients,
                bedSlopes
        );

        // --- 4. Reconstrucción del Estado ---
        reconstructStateFromGpuResults(gpuResults);
    }

    /**
     * Calcula el caudal de entrada para cada celda. El caudal de una celda 'i'
     * es igual al caudal de salida de la celda 'i-1' del paso de tiempo anterior.
     *
     * @param cellCount número de celdas del río
     * @return Array de caudales de entrada para cada celda en el instante a computar.
     */
    private float[] precomputeTargetDischarges(int cellCount) {
        float[] discharges = new float[cellCount];
        discharges[0] = (float) flowGenerator.getDischargeAt(currentTimeInSeconds);
        for (int i = 1; i < cellCount; i++) {
            double prevArea = geometry.getCrossSectionalArea(i - 1, currentState.getWaterDepthAt(i - 1));
            discharges[i] = (float) (prevArea * currentState.getVelocityAt(i - 1));
        }
        return discharges;
    }

    /**
     * Prepara las profundidades del estado actual como estimación inicial para el solver de Newton-Raphson.
     * Convierte los valores a FP32 y asegura que ninguna profundidad sea cero para evitar singularidades matemáticas.
     *
     * @param cellCount número de celdas del río.
     * @return Array de profundidades iniciales en FP32.
     */
    private float[] sanitizeInitialDepths(int cellCount) {
        float[] depths = new float[cellCount];
        for (int i = 0; i < cellCount; i++) {
            double d = currentState.getWaterDepthAt(i);
            depths[i] = (d <= 1e-3) ? 0.001f : (float) d;
        }
        return depths;
    }

    /**
     * Calcula la pendiente del lecho a partir del perfil de elevación y la sanitiza.
     * La pendiente en la celda 'i' es (elev_i - elev_{i+1}) / dx.
     * Asegura que no haya pendientes nulas o negativas que impidan el cálculo de Manning.
     *
     * @return Un array de pendientes en precisión simple, listo para la GPU.
     */
    private float[] calculateAndSanitizeBedSlopes() {
        double[] elevations = geometry.cloneElevationProfile();
        double dx = geometry.getDx();
        int cellCount = geometry.getCellCount();
        float[] sanitizedSlopes = new float[cellCount];

        for (int i = 0; i < cellCount - 1; i++) {
            double slope = (elevations[i] - elevations[i + 1]) / dx;
            sanitizedSlopes[i] = (slope <= 1e-7) ? 1e-7f : (float) slope;
        }

        if (cellCount > 1) {
            sanitizedSlopes[cellCount - 1] = sanitizedSlopes[cellCount - 2];
        } else if (cellCount == 1) {
            sanitizedSlopes[0] = 1e-7f; // Evitar error en río de una sola celda.
        }

        return sanitizedSlopes;
    }

    /**
     * Construye el nuevo objeto RiverState a partir de los resultados brutos de la GPU.
     * También calcula los nuevos valores de temperatura y pH en la CPU.
     *
     * @param gpuResults Array de floats intercalado [profundidad0, velocidad0, ...].
     */
    private void reconstructStateFromGpuResults(float[] gpuResults) {
        int cellCount = geometry.getCellCount();
        double[] newWaterDepth = new double[cellCount];
        double[] newVelocity = new double[cellCount];

        // Desempaqueta los resultados intercalados.
        for (int i = 0; i < cellCount; i++) {
            newWaterDepth[i] = gpuResults[i * 2];
            newVelocity[i] = gpuResults[i * 2 + 1];
        }

        // Temperatura y pH se siguen calculando en Java.
        double[][] tempAndPh = calculateTemperatureAndPh();
        double[] newTemperature = tempAndPh[0];
        double[] newPh = tempAndPh[1];

        this.currentState = new RiverState(newWaterDepth, newVelocity, newTemperature, newPh);
    }

    /**
     * Calcula la temperatura y el pH para cada celda del río.
     * Esta lógica se mantiene en la CPU ya que no es computacionalmente intensiva.
     *
     * @return Un array 2D donde [0] es el array de temperaturas y [1] es el de pH.
     */
    private double[][] calculateTemperatureAndPh() {
        final int cellCount = geometry.getCellCount();
        final double[] elevations = geometry.cloneElevationProfile();
        final double dx = geometry.getDx();

        double[] newTemperature = new double[cellCount];
        double[] newPh = new double[cellCount];
        double[] bottomWidths = geometry.cloneBottomWidth();
        double[] phProfile = geometry.clonePhProfile();

        // 1. Calcular la Temperatura Base (dependiente del tiempo)
        final double SECONDS_IN_A_DAY = 24.0 * 3600.0;
        final double DAYS_IN_A_YEAR = 365.25;
        final double dayOfYear = (currentTimeInSeconds / SECONDS_IN_A_DAY) % DAYS_IN_A_YEAR;
        final double secondOfDay = currentTimeInSeconds % SECONDS_IN_A_DAY;
        final double seasonalCycle = Math.sin((dayOfYear / DAYS_IN_A_YEAR) * 2.0 * Math.PI);
        final double baseSeasonalTemp = config.averageAnnualTemperature() + config.seasonalTempVariation() * seasonalCycle;
        final double dailyCycle = Math.sin((secondOfDay / SECONDS_IN_A_DAY) * 2.0 * Math.PI);
        final double baseTempForCurrentTime = baseSeasonalTemp + config.dailyTempVariation() * dailyCycle;

        // 2. Calcular la temperatura final para cada celda
        for (int i = 0; i < cellCount; i++) {
            // Componente de Gradiente Longitudinal
            double position = i * dx;
            double gradientFactor = Math.max(0, 1.0 - (position / config.headwaterCoolingDistance()));
            double headwaterCooling = -config.maxHeadwaterCoolingEffect() * gradientFactor;

            // Componente de Efecto Geomorfológico
            double relativeWidth = bottomWidths[i] / config.baseWidth();
            double widthEffect = config.widthHeatingFactor() * Math.max(0, relativeWidth - 1.0);

            double slope = (i < cellCount - 1)
                    ? (elevations[i] - elevations[i + 1]) / dx
                    : (elevations[i - 1] - elevations[i]) / dx;
            double relativeSlope = Math.max(0, slope) / config.averageSlope();
            double slopeEffect = -config.slopeCoolingFactor() * Math.max(0, relativeSlope - 1.0);

            double geomorphologyEffect = widthEffect + slopeEffect;

            // Componente de Ruido Local
            double noiseEffect = tempNoise.GetNoise(i, 0) * config.temperatureNoiseAmplitude();

            // Asignar la temperatura y pH finales
            newTemperature[i] = baseTempForCurrentTime + headwaterCooling + geomorphologyEffect + noiseEffect;
            newPh[i] = phProfile[i]; // TODO relacionar ph con otras variables
        }

        return new double[][]{newTemperature, newPh};
    }

    /**
     * Convierte un array de doubles a un array de floats.
     *
     * @param doubleArray Array de entrada en FP64.
     * @return Nuevo array en FP32.
     */
    private float[] toFloatArray(double[] doubleArray) {
        float[] floatArray = new float[doubleArray.length];
        for (int i = 0; i < doubleArray.length; i++) {
            floatArray[i] = (float) doubleArray[i];
        }
        return floatArray;
    }

    /**
     * Declaración del JNI nativo que se comunica con la librería C++/CUDA.
     * La implementación real se encuentra en el código nativo compilado.
     *
     * @param targetDischarges    Caudales de entrada para cada celda.
     * @param initialDepthGuesses Profundidades del paso anterior como estimación inicial.
     * @param bottomWidths        Anchos de fondo de cada celda.
     * @param sideSlopes          Pendientes de talud de cada celda.
     * @param manningCoefficients Coeficientes de rugosidad de Manning.
     * @param bedSlopes           Pendientes del lecho del río.
     * @return Array de floats intercalado con los resultados [profundidad0, velocidad0, ...].
     */
    protected native float[] solveManningGpu(
            float[] targetDischarges,
            float[] initialDepthGuesses,
            float[] bottomWidths,
            float[] sideSlopes,
            float[] manningCoefficients,
            float[] bedSlopes
    );

    /**
     * Devuelve el caudal de entrada generado para el tiempo actual de la simulación.
     *
     * @return Caudal de entrada en m³/s.
     */
    public double getCurrentInputDischarge() {
        return flowGenerator.getDischargeAt(currentTimeInSeconds);
    }

    /**
     * Generador de perfil de caudal variable a lo largo del tiempo usando ruido Perlin.
     * Permite simular hidrogramas realistas con variaciones suaves.
     */
    private static class FlowProfileGenerator {

        /**
         * Instancia del generador de ruido.
         */
        private final FastNoiseLite noise;
        /**
         * El caudal promedio sobre el cual se aplican las variaciones.
         */
        private final double baseDischarge;
        /**
         * La magnitud máxima de la variación sobre el caudal base.
         */
        private final double noiseAmplitude;

        /**
         * Constructor para el generador de caudal.
         *
         * @param seed           Semilla para la generación de ruido. Determina la secuencia de valores.
         * @param baseDischarge  El caudal promedio o base.
         * @param noiseAmplitude Magnitud de la variación.
         * @param noiseFrequency Frecuencia de la variación (valores bajos para cambios lentos).
         */
        public FlowProfileGenerator(int seed, double baseDischarge, double noiseAmplitude, float noiseFrequency) {
            this.noise = new FastNoiseLite(seed);
            this.noise.SetNoiseType(FastNoiseLite.NoiseType.Perlin);
            this.noise.SetFrequency(noiseFrequency);

            this.baseDischarge = baseDischarge;
            this.noiseAmplitude = noiseAmplitude;
        }

        /**
         * Calcula el valor del caudal en un instante de tiempo específico.
         *
         * @param timeInSeconds El punto en el tiempo (en segundos) para el cual se calcula el caudal.
         * @return El valor del caudal, garantizado como no negativo.
         */
        public double getDischargeAt(double timeInSeconds) {
            // Obtenemos un valor de ruido Perlin 1D en el rango [-1, 1].
            float noiseValue = noise.GetNoise((float) timeInSeconds, 0.0f);

            // Calculamos la variación y la sumamos al caudal base.
            double discharge = baseDischarge + noiseValue * noiseAmplitude;

            // Nos aseguramos de que el caudal nunca sea negativo.
            return Math.abs(discharge);
        }
    }
}