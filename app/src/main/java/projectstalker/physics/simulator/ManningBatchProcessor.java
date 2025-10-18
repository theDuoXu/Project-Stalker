package projectstalker.physics.simulator;

import lombok.extern.slf4j.Slf4j;
import projectstalker.config.SimulationConfig;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;
import projectstalker.domain.simulation.ManningSimulationResult;
import projectstalker.physics.impl.ManningProfileCalculatorTask;
import projectstalker.physics.jni.ManningGpuSolver;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.*;

/**
 * Procesa un lote (batch) de pasos de tiempo de simulación de Manning.
 * Encapsula la lógica de concurrencia (CPU) y la delegación al solver de GPU para el batch.
 */
@Slf4j
public class ManningBatchProcessor {

    private final RiverGeometry geometry;
    private final ExecutorService threadPool;
    private final int cellCount;
    private final ManningGpuSolver gpuSolver;
    /**
     * Constructor.
     *
     * @param geometry         La geometría estática del río.
     * @param simulationConfig La configuración que define el número de CPUs.
     */
    public ManningBatchProcessor(RiverGeometry geometry, SimulationConfig simulationConfig) {
        this.geometry = geometry;
        this.cellCount = geometry.getCellCount();

        int processorCount = simulationConfig.getCpuProcessorCount();
        this.threadPool = Executors.newFixedThreadPool(Math.max(processorCount, 1));
        log.info("ManningBatchProcessor inicializado con pool de {} hilos.", processorCount);
        this.gpuSolver = new ManningGpuSolver();
    }

    /**
     * Procesa un batch completo de simulación y devuelve el resultado.
     *
     * @param batchSize          El número de pasos a simular.
     * @param initialRiverState  El estado del río al inicio del batch.
     * @param allDischargeProfiles Perfiles de caudal pre-calculados para cada paso del batch.
     * @param phTmp              Arrays pre-calculados de [Temperatura, pH] para cada paso del batch.
     * @param isGpuAccelerated   Indica si se debe usar el solver de GPU o CPU concurrente.
     * @return El resultado de la simulación del batch.
     */
    public ManningSimulationResult processBatch(int batchSize,
                                                RiverState initialRiverState, double[][] allDischargeProfiles,
                                                double[][][] phTmp, boolean isGpuAccelerated) {
        long startTimeInMilis = System.currentTimeMillis();

        ManningSimulationResult result;
        if (isGpuAccelerated) {
            result = gpuComputeBatch(batchSize, initialRiverState, allDischargeProfiles, phTmp);
        } else {
            result = cpuComputeBatch(batchSize, initialRiverState, allDischargeProfiles, phTmp);
        }
        return result.withSimulationTime(System.currentTimeMillis() - startTimeInMilis);
    }

    /**
     * Construye la matriz de perfiles de caudal simulando la propagación de las ondas
     * a lo largo del batch.
     *
     * @param batchSize          El número de pasos en el batch.
     * @param newDischarges      Los caudales de entrada para cada paso.
     * @param initialDischarges  Los caudales del río justo antes del batch.
     * @return Una matriz `[batchSize][cellCount]` con los perfiles de caudal.
     */
    public double[][] createDischargeProfiles(int batchSize, double[] newDischarges, double[] initialDischarges) {
        double[][] dischargeProfiles = new double[batchSize][cellCount];
        for (int j = 0; j < batchSize; j++) {
            for (int k = 0; k < cellCount; k++) {
                if (k <= j) {
                    // La onda de caudal del tiempo (j-k) está ahora en la celda k
                    dischargeProfiles[j][k] = newDischarges[j - k];
                } else {
                    // La celda k aún no ha sido alcanzada, usa el caudal inicial (pre-batch)
                    int sourceIndex = k - (j + 1);
                    if (sourceIndex >= 0) {
                        dischargeProfiles[j][k] = initialDischarges[sourceIndex];
                    } else {
                        // Esto no debería pasar con el algoritmo actual
                        throw new IllegalStateException("Error de propagación: índice de origen negativo.");
                    }
                }
            }
        }
        return dischargeProfiles;
    }

    /**
     * Realiza el cómputo del batch utilizando el pool de hilos de la CPU.
     */
    private ManningSimulationResult cpuComputeBatch(int batchSize, RiverState initialRiverState,
                                                    double[][] allDischargeProfiles, double[][][] phTmp) {
        List<ManningProfileCalculatorTask> tasks = new ArrayList<>(batchSize);
        double[] initialWaterDepth = initialRiverState.waterDepth();

        // 1. Crear las tareas
        for (int i = 0; i < batchSize; i++) {
            tasks.add(new ManningProfileCalculatorTask(
                    allDischargeProfiles[i],
                    initialWaterDepth,
                    geometry
            ));
        }

        // 2. Ejecutar y esperar
        List<Future<ManningProfileCalculatorTask>> futures;
        try {
            futures = threadPool.invokeAll(tasks);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("El hilo de simulación fue interrumpido mientras esperaba las tareas.", e);
        }

        // 3. Recoger resultados
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

        // 4. Ensamblar y devolver
        List<RiverState> states = assembleRiverStateResults(batchSize, phTmp, resultingDepths, resultingVelocities);
        return ManningSimulationResult.builder().geometry(this.geometry).states(states).build();
    }

    /**
     * Realiza el cómputo del batch delegando al solver nativo de GPU.
     */
    private ManningSimulationResult gpuComputeBatch(int batchSize, RiverState initialRiverState,
                                                    double[][] allDischargeProfiles, double[][][] phTmp) {

        // 1. Llamada al solver de GPU
        double[][][] results = gpuSolver.solveBatch(initialRiverState.waterDepth(), allDischargeProfiles, this.geometry);

        // 2. Validación y manejo de errores
        if (results == null || results.length != batchSize) {
            log.error("Error crítico en gpuComputeBatch: El resultado del solver es nulo o no coincide con el tamaño del batch esperado. Esperado: {}, Obtenido: {}",
                    batchSize, results != null ? results.length : "null");
            return ManningSimulationResult.builder().geometry(this.geometry).states(Collections.emptyList()).build();
        }

        // 3. Desempaquetar resultados
        double[][] resultingDepths = new double[batchSize][cellCount];
        double[][] resultingVelocities = new double[batchSize][cellCount];

        for (int i = 0; i < batchSize; i++) {
            double[] depthsForStep = results[i][0];
            double[] velocitiesForStep = results[i][1];

            if (depthsForStep == null || depthsForStep.length != cellCount || velocitiesForStep == null || velocitiesForStep.length != cellCount) {
                log.error("Error crítico en gpuComputeBatch: Datos corruptos en el paso de simulación {}.", i);
                throw new RuntimeException("Error crítico en gpuComputeBatch: Datos corruptos en el paso de simulación.");
            }

            resultingDepths[i] = depthsForStep;
            resultingVelocities[i] = velocitiesForStep;
        }

        // 4. Ensamblar y devolver
        List<RiverState> states = assembleRiverStateResults(batchSize, phTmp, resultingDepths, resultingVelocities);
        return ManningSimulationResult.builder().geometry(this.geometry).states(states).build();
    }

    /**
     * Ensambla los arrays de profundidad, velocidad, temperatura y pH en una lista de RiverState.
     */
    private static List<RiverState> assembleRiverStateResults(int batchSize, double[][][] phTmp, double[][] resultingDepths, double[][] resultingVelocities) {
        List<RiverState> states = new ArrayList<>(batchSize);
        for (int i = 0; i < batchSize; i++) {
            // phTmp[i][0] es el array de Temperatura, phTmp[i][1] es el array de pH
            states.add(new RiverState(resultingDepths[i], resultingVelocities[i], phTmp[i][0], phTmp[i][1]));
        }
        return states;
    }
}