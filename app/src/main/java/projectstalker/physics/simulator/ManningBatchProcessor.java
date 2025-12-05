package projectstalker.physics.simulator;

import lombok.extern.slf4j.Slf4j;
import projectstalker.config.SimulationConfig;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;
import projectstalker.domain.simulation.DenseManningResult;
import projectstalker.domain.simulation.FlyweightManningResult;
import projectstalker.domain.simulation.ISimulationResult;
import projectstalker.physics.impl.ManningProfileCalculatorTask;
import projectstalker.physics.jni.ManningGpuSolver;
import projectstalker.config.SimulationConfig.GpuStrategy;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Procesa un lote (batch) de pasos de tiempo de simulación de Manning.
 */
@Slf4j
public class ManningBatchProcessor implements AutoCloseable {


    private final RiverGeometry geometry;
    private final ExecutorService threadPool;
    private final int cellCount;
    private final ManningGpuSolver gpuSolver;

    // Parámetro de configuración cacheado
    private final int fullEvolutionStride;

    public ManningBatchProcessor(RiverGeometry geometry, SimulationConfig simulationConfig) {
        this.geometry = geometry;
        this.cellCount = geometry.getCellCount();

        int processorCount = simulationConfig.getCpuProcessorCount();
        this.threadPool = Executors.newFixedThreadPool(Math.max(processorCount, 1));

        if (simulationConfig.isUseGpuAccelerationOnManning()) {
            log.info("Inicializando sesión GPU (Stateful DMA mode)...");
            this.gpuSolver = new ManningGpuSolver(geometry);
        } else {
            log.info("Modo solo CPU: No se inicializará la sesión GPU.");
            this.gpuSolver = null;
        }

        // Leer el stride de la configuración (Default 1 si no está definido)
        // Asumiendo que el getter se llama getGpuFullEvolutionStride()
        this.fullEvolutionStride = Math.max(1, simulationConfig.getGpuFullEvolutionStride());

        log.info("ManningBatchProcessor inicializado. (CPU Threads: {}, GPU Stride: {})", processorCount, fullEvolutionStride);
    }

    public ISimulationResult processBatch(int batchSize,
                                          RiverState initialRiverState,
                                          float[] newInflows,
                                          float[][][] phTmp,
                                          boolean isGpuAccelerated,
                                          GpuStrategy strategy) {

        if (isGpuAccelerated) {
            return gpuComputeBatch(batchSize, initialRiverState, newInflows, phTmp, strategy);
        } else {
            return cpuComputeBatch(batchSize, initialRiverState, newInflows, phTmp);
        }
    }

    public ISimulationResult processBatch(int batchSize,
                                          RiverState initialRiverState,
                                          float[] newInflows,
                                          float[][][] phTmp,
                                          boolean isGpuAccelerated) {
        // Por defecto SAFE
        return processBatch(batchSize, initialRiverState, newInflows, phTmp, isGpuAccelerated, GpuStrategy.SMART_SAFE);
    }

    private ISimulationResult gpuComputeBatch(int batchSize, RiverState initialRiverState,
                                              float[] newInflows, float[][][] phTmp,
                                              GpuStrategy strategy) {
        if (this.gpuSolver == null) {
            throw new IllegalStateException("Se solicitó ejecución GPU, pero el procesador se inicializó en modo CPU.");
        }

        try {
            long t0 = System.nanoTime();

            float[] initialQ = initialRiverState.discharge(this.geometry);
            float[] initialDepths = initialRiverState.waterDepth();

            long t1 = System.nanoTime();

            float[][][] packedResults;
            boolean executedAsFull = false;

            switch (strategy) {
                case FULL_EVOLUTION:
                    // PASAMOS EL STRIDE CONFIGURADO
                    packedResults = gpuSolver.solveFullEvolutionBatch(initialDepths, newInflows, initialQ, this.fullEvolutionStride);
                    executedAsFull = true;
                    break;

                case SMART_TRUSTED:
                    packedResults = gpuSolver.solveSmartBatch(initialDepths, newInflows, initialQ, true);
                    break;

                case SMART_SAFE:
                default:
                    try {
                        packedResults = gpuSolver.solveSmartBatch(initialDepths, newInflows, initialQ, false);
                    } catch (IllegalStateException e) {
                        log.warn("GPU Smart Optimization rechazada. Fallback a FULL EVOLUTION (Stride={}). Motivo: {}", this.fullEvolutionStride, e.getMessage());
                        // EN FALLBACK TAMBIÉN USAMOS EL STRIDE
                        packedResults = gpuSolver.solveFullEvolutionBatch(initialDepths, newInflows, initialQ, this.fullEvolutionStride);
                        executedAsFull = true;
                    }
                    break;
            }

            long t2 = System.nanoTime();

            // Validación de integridad básica del retorno
            // Nota: En modo FULL con Stride > 1, packedResults.length (tiempo) será menor que batchSize.
            // FlyweightManningResult debe saber manejar esto o debemos expandirlo aquí si fuera necesario.
            // Como ISimulationResult abstrae el tiempo, FlyweightManningResult usará lo que tenga.
            if (packedResults == null || packedResults.length == 0) {
                log.error("Error crítico GPU: Resultado vacío.");
                return DenseManningResult.builder().geometry(this.geometry).states(Collections.emptyList()).simulationTime(0).build();
            }

            long simulationTimeMs = (t2 - t0) / 1_000_000;
            int effectiveStride = (executedAsFull) ? this.fullEvolutionStride : 1;

            ISimulationResult result = new FlyweightManningResult(
                    this.geometry,
                    simulationTimeMs,
                    initialRiverState,
                    packedResults,
                    phTmp,
                    effectiveStride
            );

            long t4 = System.nanoTime();

            // LOG
            double totalMs = (t4 - t0) / 1e6;
            if (totalMs > 5.0) {
                double prepMs = (t1 - t0) / 1e6;
                double gpuMs = (t2 - t1) / 1e6;
                double wrapMs = (t4 - t2) / 1e6;
                String modeLabel = executedAsFull ? ("FULL(s=" + this.fullEvolutionStride + ")") : "SMART";

                log.debug(">>> GPU DMA BATCH [{}]: Total={}ms | Prep={}ms | GPU+DMA={}ms | Wrap={}ms <<<",
                        modeLabel,
                        String.format("%.2f", totalMs),
                        String.format("%.2f", prepMs),
                        String.format("%.2f", gpuMs),
                        String.format("%.2f", wrapMs)
                );
            }

            return result;

        } catch (Exception e) {
            log.error("Fallo crítico en ejecución GPU (BatchSize: {})", batchSize, e);
            throw new RuntimeException("Error en simulación GPU Manning", e);
        }
    }

    // --- CPU Compute Batch ---
    private ISimulationResult cpuComputeBatch(int batchSize, RiverState initialRiverState,
                                              float[] newInflows, float[][][] phTmp) {
        long tStart = System.currentTimeMillis();
        float[][] allDischargeProfiles = createDischargeProfiles(batchSize, newInflows, initialRiverState.discharge(this.geometry));
        List<ManningProfileCalculatorTask> tasks = new ArrayList<>(batchSize);
        float[] initialWaterDepth = initialRiverState.waterDepth();

        for (int i = 0; i < batchSize; i++) {
            tasks.add(new ManningProfileCalculatorTask(allDischargeProfiles[i], initialWaterDepth, geometry));
        }

        List<Future<ManningProfileCalculatorTask>> futures;
        try {
            futures = threadPool.invokeAll(tasks);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Interrupción en CPU batch.", e);
        }

        float[][] resultingDepths = new float[batchSize][cellCount];
        float[][] resultingVelocities = new float[batchSize][cellCount];
        for (int i = 0; i < batchSize; i++) {
            try {
                ManningProfileCalculatorTask completedTask = futures.get(i).get();
                System.arraycopy(completedTask.getCalculatedWaterDepth(), 0, resultingDepths[i], 0, cellCount);
                System.arraycopy(completedTask.getCalculatedVelocity(), 0, resultingVelocities[i], 0, cellCount);
            } catch (Exception e) {
                throw new RuntimeException("Fallo en tarea CPU #" + i, e);
            }
        }
        List<RiverState> states = assembleRiverStateResults(batchSize, phTmp, resultingDepths, resultingVelocities);
        long tEnd = System.currentTimeMillis();
        return DenseManningResult.builder().geometry(this.geometry).states(states).simulationTime(tEnd - tStart).build();
    }

    private float[][] createDischargeProfiles(int batchSize, float[] newDischarges, float[] initialDischarges) {
        float[][] dischargeProfiles = new float[batchSize][cellCount];
        for (int j = 0; j < batchSize; j++) {
            for (int k = 0; k < cellCount; k++) {
                if (k <= j) {
                    dischargeProfiles[j][k] = newDischarges[j - k];
                } else {
                    int sourceIndex = k - (j + 1);
                    if (sourceIndex >= 0) {
                        dischargeProfiles[j][k] = initialDischarges[sourceIndex];
                    } else {
                        dischargeProfiles[j][k] = initialDischarges[0];
                    }
                }
            }
        }
        return dischargeProfiles;
    }

    private static List<RiverState> assembleRiverStateResults(int batchSize, float[][][] phTmp, float[][] resultingDepths, float[][] resultingVelocities) {
        List<RiverState> states = new ArrayList<>(batchSize);
        for (int i = 0; i < batchSize; i++) {
            states.add(RiverState.builder()
                    .waterDepth(resultingDepths[i])
                    .velocity(resultingVelocities[i])
                    .temperature(phTmp[i][0])
                    .ph(phTmp[i][1])
                    .contaminantConcentration(new float[resultingDepths[i].length])
                    .build());
        }
        return states;
    }

    @Override
    public void close() {
        if (gpuSolver != null) gpuSolver.close();
        if (threadPool != null && !threadPool.isShutdown()) threadPool.shutdown();
        log.info("ManningBatchProcessor cerrado.");
    }
}