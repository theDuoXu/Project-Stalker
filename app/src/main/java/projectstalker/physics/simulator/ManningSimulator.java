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
import projectstalker.physics.impl.ManningProfileCalculatorTask;
import projectstalker.physics.impl.SequentialManningHydrologySolver;
import projectstalker.physics.jni.ManningGpuSolver;
import projectstalker.physics.model.FlowProfileModel;
import projectstalker.physics.model.RiverTemperatureModel;
import projectstalker.physics.i.IHydrologySolver;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;

/**
 * Orquesta la simulación hidrológica del río, delegando los cálculos
 * a solvers específicos de CPU o GPU.
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
    private final ExecutorService threadPool;

    @Getter
    @Setter
    private int processorCount;

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

    // --- Métricas de Rendimiento ---
    @Getter
    private long cpuFillTimeNanos = 0;
    @Getter
    private int cpuFillIterations = 0;

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

        int cellCount = geometry.getCellCount();
        this.currentState = new RiverState(new double[cellCount], new double[cellCount], new double[cellCount], new double[cellCount]);
        this.currentTimeInSeconds = 0.0;
        this.isGpuAccelerated = simulationConfig.isUseGpuAccelerationOnManning();

        this.processorCount = simulationConfig.getCpuProcessorCount();
        this.threadPool = Executors.newFixedThreadPool(Math.max(processorCount, 1));
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
     * Avanza la simulación para un batch de pasos de tiempo de forma concurrente en la CPU.
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

        // 1. Pre-cómputo de caudales de entrada y otros datos para todo el batch.
        double[] newDischarges = new double[batchSize];
        double[][][] phTmp = new double[batchSize][2][cellCount];
        for (int i = 0; i < batchSize; i++) {
            newDischarges[i] = flowGenerator.getDischargeAt(currentTimeInSeconds);
            phTmp[i] = calculateTemperatureAndPh();
            currentTimeInSeconds += deltaTimeInSeconds;
        }

        // 2. Calcular el estado de caudales en el río justo antes de que entre la nueva onda.
        double[] initialDischarges = new double[cellCount];
        for (int j = 0; j < cellCount - 1; j++) {
            double area = geometry.getCrossSectionalArea(j, currentState.getWaterDepthAt(j));
            double velocity = currentState.getVelocityAt(j);
            initialDischarges[j + 1] = area * velocity;
        }

        // 3. Construir la matriz de perfiles de caudal para cada paso de tiempo del batch.
        double[][] allDischargeProfiles = createDischargeProfiles(batchSize, cellCount, newDischarges, initialDischarges);

        // 4. Crear las tareas de cálculo para cada perfil de caudal.
        if (isGpuAccelerated) {
            return gpuComputeBatch(batchSize, cellCount, allDischargeProfiles, phTmp);
        } else {
            return cpuComputeBatch(batchSize, cellCount, allDischargeProfiles, phTmp);
        }

    }


    private ManningSimulationResult cpuComputeBatch(int batchSize, int cellCount, double[][] allDischargeProfiles, double[][][] phTmp) {
        List<ManningProfileCalculatorTask> tasks = new ArrayList<>(batchSize);
        for (int i = 0; i < batchSize; i++) {
            tasks.add(new ManningProfileCalculatorTask(
                    allDischargeProfiles[i],
                    currentState.waterDepth(),
                    geometry
            ));
        }

        // Enviar todas las tareas y esperar de forma bloqueante a que terminen.
        List<Future<ManningProfileCalculatorTask>> futures;
        try {
            futures = threadPool.invokeAll(tasks);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("El hilo de simulación fue interrumpido mientras esperaba las tareas.", e);
        }

        // Recoger y ensamblar los resultados de todas las tareas completadas.
        double[][] resultingDepths = new double[batchSize][cellCount];
        double[][] resultingVelocities = new double[batchSize][cellCount];
        for (int i = 0; i < batchSize; i++) {
            try {
                ManningProfileCalculatorTask completedTask = futures.get(i).get();
                System.arraycopy(completedTask.getCalculatedWaterDepth(), 0, resultingDepths[i], 0, cellCount);
                System.arraycopy(completedTask.getCalculatedVelocity(), 0, resultingVelocities[i], 0, cellCount);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException("La recolección de resultados fue interrumpida.", e);
            } catch (ExecutionException e) {
                throw new RuntimeException("La tarea de cálculo #" + i + " falló.", e.getCause());
            }
        }

        // 7. Construir y devolver el objeto de resultado final.
        List<RiverState> states = new ArrayList<>(batchSize);
        for (int i = 0; i < batchSize; i++) {
            states.add(new RiverState(resultingDepths[i], resultingVelocities[i], phTmp[i][0], phTmp[i][1]));
        }

        return ManningSimulationResult.builder().geometry(this.geometry).states(states).build();
    }

    private ManningSimulationResult gpuComputeBatch(int batchSize, int cellCount, double[][] allDischargeProfiles, double[][][] phTmp) {
        return null;
    }


    /**
     * Construye la matriz de perfiles de caudal simulando la propagación de las ondas.
     */
    private double[][] createDischargeProfiles(int batchSize, int cellCount, double[] newDischarges, double[] initialDischarges) {
        double[][] dischargeProfiles = new double[batchSize][cellCount];
        for (int j = 0; j < batchSize; j++) {
            for (int k = 0; k < cellCount; k++) {
                if (k <= j) {
                    dischargeProfiles[j][k] = newDischarges[j - k];
                } else {
                    int sourceIndex = k - (j + 1);
                    if (sourceIndex >= 0) {
                        dischargeProfiles[j][k] = initialDischarges[sourceIndex];
                    } else {
                        throw new IllegalStateException("Se calculó un índice de origen negativo (" + sourceIndex + ") para j=" + j + " y k=" + k + ". Error en el algoritmo de propagación.");
                    }
                }
            }
        }
        return dischargeProfiles;
    }

    /**
     * Ejecuta un paso de simulación usando el solver secuencial de Java (CPU).
     */
    private void runCpuStep() {
        long startTime = System.nanoTime();
        double inputDischarge = flowGenerator.getDischargeAt(currentTimeInSeconds);
        this.currentState = cpuSolver.calculateNextState(currentState, geometry, config, currentTimeInSeconds, inputDischarge);
        this.cpuFillIterations++;
        this.cpuFillTimeNanos += (System.nanoTime() - startTime);
    }

    /**
     * Orquesta un paso de simulación delegando el cálculo completo al solver de GPU.
     */
    private void runGpuStep() {
        // 1. Delegar el cálculo hidrológico completo al solver de GPU.
        double[][] gpuResults = ManningGpuSolver.solve(currentState, geometry, flowGenerator, currentTimeInSeconds);
        double[] newWaterDepth = gpuResults[0];
        double[] newVelocity = gpuResults[1];

        // 2. Temperatura y pH se siguen calculando en la CPU.
        double[][] tempAndPh = calculateTemperatureAndPh();
        double[] newTemperature = tempAndPh[0];
        double[] newPh = tempAndPh[1];

        // 3. Construir el nuevo estado del río a partir de todos los resultados.
        this.currentState = new RiverState(newWaterDepth, newVelocity, newTemperature, newPh);
    }

    /**
     * Calcula los perfiles de temperatura y pH para el estado actual.
     */
    private double[][] calculateTemperatureAndPh() {
        double[] newTemperature = temperatureModel.calculate(currentTimeInSeconds);
        double[] newPh = geometry.clonePhProfile();
        return new double[][]{newTemperature, newPh};
    }

    /**
     * Devuelve el caudal de entrada para el tiempo actual de la simulación.
     */
    public double getCurrentInputDischarge() {
        return flowGenerator.getDischargeAt(currentTimeInSeconds);
    }
}