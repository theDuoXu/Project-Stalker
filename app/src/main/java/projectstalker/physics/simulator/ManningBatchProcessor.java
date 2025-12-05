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
import java.util.concurrent.*;

/**
 * Procesa un lote (batch) de pasos de tiempo de simulación de Manning.
 * <p>
 * Implementa un enfoque híbrido y actúa como Factoría de Resultados:
 * 1. GPU (Stateful + DMA): Delega al {@link ManningGpuSolver}. Soporta modos SMART (Optimized) y FULL (Robust).
 * 2. CPU (Concurrent): Fallback a threads de Java con matrices densas.
 */
@Slf4j
public class ManningBatchProcessor implements AutoCloseable {

    private final RiverGeometry geometry;
    private final ExecutorService threadPool;
    private final int cellCount;

    // Stateful Solver: Encapsula la sesión GPU y los buffers DMA persistentes.
    private final ManningGpuSolver gpuSolver;

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

        log.info("ManningBatchProcessor inicializado. (CPU Threads: {})", processorCount);
    }

    /**
     * Procesa un batch completo.
     * @param strategy La estrategia GPU a utilizar (Trusted, Safe o Full).
     */
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

    // Sobrecarga para compatibilidad hacia atrás (default trust=false -> Safe Mode)
    public ISimulationResult processBatch(int batchSize,
                                          RiverState initialRiverState,
                                          float[] newInflows,
                                          float[][][] phTmp,
                                          boolean isGpuAccelerated) {
        return processBatch(batchSize, initialRiverState, newInflows, phTmp, isGpuAccelerated, GpuStrategy.SMART_SAFE);
    }

    /**
     * Realiza el cómputo GPU con selección explícita de estrategia.
     */
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

            // EJECUCIÓN NATIVA CON SELECTOR DE ESTRATEGIA
            float[][][] packedResults;
            boolean executedAsFull = false;

            switch (strategy) {
                case FULL_EVOLUTION:
                    // CAMINO DIRECTO 1: Ejecución Robusta Explícita
                    packedResults = gpuSolver.solveFullEvolutionBatch(initialDepths, newInflows, initialQ);
                    executedAsFull = true;
                    break;

                case SMART_TRUSTED:
                    // CAMINO DIRECTO 2: Ejecución Optimizada "Ciega" (Confianza total)
                    packedResults = gpuSolver.solveSmartBatch(initialDepths, newInflows, initialQ, true);
                    break;

                case SMART_SAFE:
                default:
                    // CAMINO HÍBRIDO: Intenta optimizar, pero con red de seguridad (Try-Catch)
                    try {
                        // trust=false fuerza la validación interna en el solver
                        packedResults = gpuSolver.solveSmartBatch(initialDepths, newInflows, initialQ, false);
                    } catch (IllegalStateException e) {
                        log.warn("GPU Smart Optimization rechazada (Safety Check). Fallback a FULL EVOLUTION. Motivo: {}", e.getMessage());
                        packedResults = gpuSolver.solveFullEvolutionBatch(initialDepths, newInflows, initialQ);
                        executedAsFull = true;
                    }
                    break;
            }

            long t2 = System.nanoTime();

            if (packedResults == null || packedResults.length != batchSize) {
                log.error("Error crítico GPU: BatchSize incorrecto en retorno.");
                return DenseManningResult.builder()
                        .geometry(this.geometry)
                        .states(Collections.emptyList())
                        .simulationTime(0)
                        .build();
            }

            long simulationTimeMs = (t2 - t0) / 1_000_000;

            ISimulationResult result = new FlyweightManningResult(
                    this.geometry,
                    simulationTimeMs,
                    initialRiverState,
                    packedResults,
                    phTmp
            );

            long t4 = System.nanoTime();

            // LOG DE PERFILADO
            double totalMs = (t4 - t0) / 1e6;
            if (totalMs > 5.0) {
                double prepMs = (t1 - t0) / 1e6;
                double gpuMs = (t2 - t1) / 1e6;
                double wrapMs = (t4 - t2) / 1e6;

                String modeLabel = executedAsFull ? "FULL" : "SMART";

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

    // --- CPU Compute Batch (Legacy) ---
    private ISimulationResult cpuComputeBatch(int batchSize, RiverState initialRiverState,
                                              float[] newInflows, float[][][] phTmp) {
        long tStart = System.currentTimeMillis();

        float[][] allDischargeProfiles = createDischargeProfiles(batchSize, newInflows, initialRiverState.discharge(this.geometry));

        List<ManningProfileCalculatorTask> tasks = new ArrayList<>(batchSize);
        float[] initialWaterDepth = initialRiverState.waterDepth();

        for (int i = 0; i < batchSize; i++) {
            tasks.add(new ManningProfileCalculatorTask(
                    allDischargeProfiles[i],
                    initialWaterDepth,
                    geometry
            ));
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

        return DenseManningResult.builder()
                .geometry(this.geometry)
                .states(states)
                .simulationTime(tEnd - tStart)
                .build();
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
        if (gpuSolver != null) {
            gpuSolver.close();
        }
        if (threadPool != null && !threadPool.isShutdown()) {
            threadPool.shutdown();
        }
        log.info("ManningBatchProcessor cerrado.");
    }
}