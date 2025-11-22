package projectstalker.physics.simulator;

import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import projectstalker.config.RiverConfig;
import projectstalker.config.SimulationConfig;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;
import projectstalker.domain.simulation.ManningSimulationResult;
import projectstalker.factory.RiverGeometryFactory;
import projectstalker.physics.impl.SequentialManningHydrologySolver;
import projectstalker.physics.jni.ManningGpuSolver;
import projectstalker.physics.jni.NativeManningGpuSingleton;
import projectstalker.physics.model.FlowProfileModel;
import projectstalker.physics.model.RiverPhModel;
import projectstalker.physics.model.RiverTemperatureModel;
import projectstalker.physics.i.IHydrologySolver;

import java.util.ArrayList;
import java.util.List;

/**
 * Orquesta la simulación hidrológica del río.
 * Enfocada en la gestión del ciclo de vida (estado, tiempo) y la delegación
 * a solvers específicos (paso a paso) o a un procesador de batch.
 */
@Slf4j
public class ManningSimulator {

    // --- Miembros de la Simulación ---
    private final RiverConfig config;
    @Getter
    private final RiverGeometry geometry;
    private final IHydrologySolver cpuSolver;
    private final FlowProfileModel flowGenerator;
    private final RiverTemperatureModel temperatureModel;
    private final RiverPhModel phModel; // Nuevo miembro
    private final ManningBatchProcessor batchProcessor; // Nuevo delegado

    // --- Estado de la Simulación ---
    @Getter
    @Setter
    private RiverState currentState;
    @Getter
    @Setter
    private double currentTimeInSeconds;
    @Getter
    @Setter
    private boolean isGpuAccelerated;

    // --- Métricas de Rendimiento (Mantenidas para el modo de paso único) ---
    @Getter
    private long cpuFillTimeNanos = 0;
    @Getter
    private int cpuFillIterations = 0;

    private final ManningGpuSolver gpuSolver;

    /**
     * Constructor del simulador.
     *
     * @param config           La configuración global para el río.
     * @param simulationConfig La configuración para la ejecución de la simulación.
     */
    public ManningSimulator(RiverConfig config, SimulationConfig simulationConfig) {
        this.config = config;
        this.geometry = new RiverGeometryFactory().createRealisticRiver(config);
        this.cpuSolver = new SequentialManningHydrologySolver();
        this.flowGenerator = new FlowProfileModel((int) config.seed(), simulationConfig.getFlowConfig());
        this.temperatureModel = new RiverTemperatureModel(config, this.geometry);
        this.phModel = new RiverPhModel(this.geometry); // Inicialización del nuevo modelo de pH

        int cellCount = geometry.getCellCount();
        this.currentState = new RiverState(new float[cellCount], new float[cellCount], new float[cellCount], new float[cellCount], new float[cellCount]);
        this.currentTimeInSeconds = 0.0;
        this.isGpuAccelerated = simulationConfig.isUseGpuAccelerationOnManning();

        // Inicialización del BatchProcessor
        this.batchProcessor = new ManningBatchProcessor(this.geometry, simulationConfig);

        if (this.isGpuAccelerated) {
            this.gpuSolver = new ManningGpuSolver();
        } else {
            this.gpuSolver = null;
        }

        log.info("ManningSimulator inicializado correctamente.");
    }

    /**
     * Avanza la simulación un único paso de tiempo.
     *
     * @param deltaTimeInSeconds El incremento de tiempo en segundos.
     */
    public void advanceTimeStep(double deltaTimeInSeconds) {
        if (isGpuAccelerated) {
            runGpuStep();
        } else {
            runCpuStep();
        }
        this.currentTimeInSeconds += deltaTimeInSeconds;
    }

    /**
     * Avanza la simulación para un batch de pasos de tiempo de forma concurrente.
     *
     * DELEGACIÓN COMPLETA AL MANNINGBATCHPROCESSOR.
     *
     * @param deltaTimeInSeconds El incremento de tiempo para cada paso del batch.
     * @param batchSize          El número de pasos a simular.
     * @return El resultado de la simulación para el batch completo.
     */
    public ManningSimulationResult advanceBatchTimeStep(double deltaTimeInSeconds, int batchSize) {
        if (batchSize <= 0) {
            throw new IllegalArgumentException("El tamaño del batch debe ser mayor que 0.");
        }

        int cellCount = geometry.getCellCount();
        double initialBatchTime = currentTimeInSeconds;
        RiverState initialRiverState = this.currentState;

        // 1. Pre-cómputo de caudales de entrada y variables fisicoquímicas para todo el batch.
        float[] newDischarges = new float[batchSize];
        float[][][] phTmp = new float[batchSize][2][cellCount]; // [i][0]=Temperatura, [i][1]=pH
        for (int i = 0; i < batchSize; i++) {
            newDischarges[i] = flowGenerator.getDischargeAt(currentTimeInSeconds);
            phTmp[i] = calculateTemperatureAndPh();
            // Avanzamos el tiempo de la simulación para el pre-cómputo de la onda de caudal
            currentTimeInSeconds += deltaTimeInSeconds;
        }

        // 2. Calcular el estado de caudales en el río justo antes de que entre la nueva onda.
        float[] initialDischarges = new float[cellCount];
        for (int j = 0; j < cellCount - 1; j++) {
            double area = geometry.getCrossSectionalArea(j, initialRiverState.getWaterDepthAt(j));
            double velocity = initialRiverState.getVelocityAt(j);
            initialDischarges[j + 1] = (float) (area * velocity);
        }

        // 3. Construir la matriz de perfiles de caudal para cada paso de tiempo del batch.
        float[][] allDischargeProfiles = batchProcessor.createDischargeProfiles(batchSize, newDischarges, initialDischarges);

        // 4. Ejecución delegada (CPU concurrente o GPU)
        ManningSimulationResult result = batchProcessor.processBatch(batchSize, initialRiverState,
                allDischargeProfiles, phTmp, isGpuAccelerated);

        // 5. Actualizar el estado final del ManningSimulator
        if (!result.getStates().isEmpty()) {
            this.currentState = result.getStates().get(batchSize - 1);
        }

        return result;
    }

    /**
     * Ejecuta un paso de simulación usando el solver secuencial de Java (CPU).
     */
    private void runCpuStep() {
        long startTime = System.nanoTime();
        double inputDischarge = flowGenerator.getDischargeAt(currentTimeInSeconds);
        // El solver CPU secuencial solo calcula [Depth, Velocity]
        RiverState nextStateHydro = cpuSolver.calculateNextState(currentState, geometry, config, currentTimeInSeconds, inputDischarge);

        // Se añaden los perfiles fisicoquímicos
        float[][] tempAndPh = calculateTemperatureAndPh();
        float[] newTemperature = tempAndPh[0];
        float[] newPh = tempAndPh[1];

        this.currentState = new RiverState(nextStateHydro.waterDepth(), nextStateHydro.velocity(), newTemperature, newPh, new float[nextStateHydro.waterDepth().length]);
        this.cpuFillIterations++;
        this.cpuFillTimeNanos += (System.nanoTime() - startTime);
    }

    /**
     * Orquesta un paso de simulación delegando el cálculo completo al solver de GPU.
     */
    private void runGpuStep() {
        if (this.gpuSolver == null) {
            throw new IllegalStateException("Modo GPU activado, pero el GpuSolver no fue inicializado.");
        }

        // 1. Delegar el cálculo hidrológico completo al solver de GPU.
        float[][] gpuResults = this.gpuSolver.solve(currentState, geometry, flowGenerator, currentTimeInSeconds);
        float[] newWaterDepth = gpuResults[0];
        float[] newVelocity = gpuResults[1];

        // 2. Temperatura y pH se siguen calculando en la CPU.
        float[][] tempAndPh = calculateTemperatureAndPh();
        float[] newTemperature = tempAndPh[0];
        float[] newPh = tempAndPh[1];

        // 3. Construir el nuevo estado del río a partir de todos los resultados.
        this.currentState = new RiverState(newWaterDepth, newVelocity, newTemperature, newPh, new float[gpuResults[0].length]);
    }

    /**
     * Calcula los perfiles de temperatura y pH para el estado actual.
     * Solo llama a los modelos específicos.
     */
    private float[][] calculateTemperatureAndPh() {
        float[] newTemperature = temperatureModel.calculate(currentTimeInSeconds);
        float[] newPh = phModel.getPhProfile();
        return new float[][]{newTemperature, newPh};
    }

    /**
     * Devuelve el caudal de entrada para el tiempo actual de la simulación.
     */
    public float getCurrentInputDischarge() {
        return flowGenerator.getDischargeAt(currentTimeInSeconds);
    }
}