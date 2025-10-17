package projectstalker.physics.simulator;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import projectstalker.config.RiverConfig;
import projectstalker.config.SimulationConfig;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;
import projectstalker.factory.RiverGeometryFactory;
import projectstalker.physics.impl.ManningHydrologySolver;
import projectstalker.physics.model.FlowProfileModel;
import projectstalker.physics.model.RiverTemperatureModel;
import projectstalker.physics.solver.IHydrologySolver;

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
                log.error("FATAL: Native library 'manning_solver' failed to load. " + "Ensure the library is in the java.library.path.", e);
                System.exit(1);
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
    private final FlowProfileModel flowGenerator;
    /**
     * Modelo dedicado al cálculo del perfil de temperaturas del río.
     */
    private final RiverTemperatureModel temperatureModel;

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
    public ManningSimulator(RiverConfig config, SimulationConfig simulationConfig) {
        this.config = config;
        this.geometry = new RiverGeometryFactory().createRealisticRiver(config);
        this.cpuSolver = new ManningHydrologySolver();
        this.flowGenerator = new FlowProfileModel((int) config.seed(), simulationConfig.getFlowConfig());

        this.temperatureModel = new RiverTemperatureModel(config, this.geometry);

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
        float[] gpuResults = solveManningGpu(targetDischarges, initialDepthGuesses, bottomWidths, sideSlopes, manningCoefficients, bedSlopes);

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
     * La lógica de temperatura se delega a RiverTemperatureModel.
     * La lógica de pH se mantiene aquí por su simplicidad.
     *
     * @return Un array 2D donde [0] es el array de temperaturas y [1] es el de pH.
     */
    private double[][] calculateTemperatureAndPh() {
        // 1. Delegar el cálculo complejo de la temperatura al modelo especializado.
        //    El modelo ya tiene la configuración, geometría y su propio generador de ruido.
        double[] newTemperature = temperatureModel.calculate(currentTimeInSeconds);

        // 2. El cálculo del pH sigue siendo simple y se puede mantener aquí.
        //    Clonamos directamente el perfil de pH de la geometría.
        double[] newPh = geometry.clonePhProfile(); // TODO: relacionar ph con otras variables

        // 3. Devolver ambos perfiles en el formato esperado.
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
    protected native float[] solveManningGpu(float[] targetDischarges, float[] initialDepthGuesses, float[] bottomWidths, float[] sideSlopes, float[] manningCoefficients, float[] bedSlopes);

    /**
     * Devuelve el caudal de entrada generado para el tiempo actual de la simulación.
     *
     * @return Caudal de entrada en m³/s.
     */
    public double getCurrentInputDischarge() {
        return flowGenerator.getDischargeAt(currentTimeInSeconds);
    }

}