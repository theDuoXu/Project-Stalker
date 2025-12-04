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

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.*;

/**
 * Procesa un lote (batch) de pasos de tiempo de simulación de Manning.
 * <p>
 * Implementa un enfoque híbrido y actúa como Factoría de Resultados:
 * 1. GPU (Stateful + DMA): Delega al {@link ManningGpuSolver} que gestiona memoria Pinned
 * para transferencias Zero-Copy. Devuelve un {@link FlyweightManningResult}.
 * 2. CPU (Concurrent): Fallback a threads de Java con matrices densas.
 * <p>
 * Implementa AutoCloseable para garantizar la liberación de la sesión JNI.
 */
@Slf4j
public class ManningBatchProcessor implements AutoCloseable {

    private final RiverGeometry geometry;
    private final ExecutorService threadPool;
    private final int cellCount;

    // Stateful Solver: Encapsula la sesión GPU y los buffers DMA persistentes.
    private final ManningGpuSolver gpuSolver;

    /**
     * Constructor.
     * Inicializa el Pool de CPU y, OPCIONALMENTE, la Sesión de GPU.
     *
     * @param geometry         La geometría estática del río.
     * @param simulationConfig La configuración para decidir si inicializar la GPU.
     */
    public ManningBatchProcessor(RiverGeometry geometry, SimulationConfig simulationConfig) {
        this.geometry = geometry;
        this.cellCount = geometry.getCellCount();

        // 1. Inicializar CPU Pool (Siempre disponible como fallback)
        int processorCount = simulationConfig.getCpuProcessorCount();
        this.threadPool = Executors.newFixedThreadPool(Math.max(processorCount, 1));

        // 2. Inicializar GPU Session (Solo si está habilitado en config)
        if (simulationConfig.isUseGpuAccelerationOnManning()) {
            log.info("Inicializando sesión GPU (Stateful DMA mode)...");
            // El solver no reserva buffers pesados hasta el primer batch (Lazy Init)
            this.gpuSolver = new ManningGpuSolver(geometry);
        } else {
            log.info("Modo solo CPU: No se inicializará la sesión GPU.");
            this.gpuSolver = null;
        }

        log.info("ManningBatchProcessor inicializado. (CPU Threads: {})", processorCount);
    }

    /**
     * Procesa un batch completo de simulación.
     * Selecciona dinámicamente la estrategia de cómputo.
     */
    public ISimulationResult processBatch(int batchSize,
                                          RiverState initialRiverState,
                                          float[] newInflows,
                                          float[][][] phTmp,
                                          boolean isGpuAccelerated) {

        if (isGpuAccelerated) {
            return gpuComputeBatch(batchSize, initialRiverState, newInflows, phTmp);
        } else {
            return cpuComputeBatch(batchSize, initialRiverState, newInflows, phTmp);
        }
    }

    /**
     * Realiza el cómputo del batch delegando al solver nativo de GPU con soporte DMA.
     * Devuelve un resultado ligero (Flyweight) para evitar saturar la RAM de Java.
     */
    private ISimulationResult gpuComputeBatch(int batchSize, RiverState initialRiverState,
                                              float[] newInflows, float[][][] phTmp) {
        if (this.gpuSolver == null) {
            throw new IllegalStateException("Se solicitó ejecución GPU, pero el procesador se inicializó en modo CPU.");
        }

        try {
            // T0: Inicio preparación de datos Java
            long t0 = System.nanoTime();

            // Extraemos arrays primitivos del dominio.
            // NOTA: 'initialDepths' y 'initialQ' son críticos para el "Init Lazy" de la sesión GPU (primer batch).
            // En batches subsiguientes, el solver los ignora porque ya tiene el estado cargado,
            // pero debemos pasarlos para satisfacer la API.
            float[] initialQ = initialRiverState.discharge(this.geometry);
            float[] initialDepths = initialRiverState.waterDepth();

            // T1: Inicio Frontera Nativa
            long t1 = System.nanoTime();

            // EJECUCIÓN NATIVA (DMA / Zero-Copy)
            // 1. Java escribe en DirectBuffer (Input).
            // 2. GPU lee vía PCIe (DMA).
            // 3. GPU escribe en DirectBuffer (Output).
            // 4. Java lee de DirectBuffer.
            // Devuelve la matriz compacta [Batch][Var][ActiveWidth]
            float[][][] packedResults = gpuSolver.solveBatch(initialDepths, newInflows, initialQ);

            // T2: Fin Frontera Nativa
            long t2 = System.nanoTime();

            if (packedResults == null || packedResults.length != batchSize) {
                log.error("Error crítico GPU: BatchSize incorrecto en retorno.");
                // Fallback de emergencia: Retornar resultado vacío seguro
                return DenseManningResult.builder()
                        .geometry(this.geometry)
                        .states(Collections.emptyList())
                        .simulationTime(0)
                        .build();
            }

            // T3: Construcción del Flyweight (Virtualización)
            // Encapsulamos los arrays crudos sin procesarlos. El desempaquetado ocurre bajo demanda (Lazy).
            long simulationTimeMs = (t2 - t0) / 1_000_000;

            ISimulationResult result = new FlyweightManningResult(
                    this.geometry,
                    simulationTimeMs,
                    initialRiverState, // Intrinsic State (Base del río)
                    packedResults,     // Extrinsic State (Delta calculado por GPU)
                    phTmp              // Datos auxiliares (Temp/Ph) passthrough
            );

            // T4: Fin total
            long t4 = System.nanoTime();

            // LOG DE PERFILADO (Solo si es lento, > 5ms)
            double totalMs = (t4 - t0) / 1e6;
            if (totalMs > 5.0) {
                double prepMs = (t1 - t0) / 1e6; // Extracción de datos Java
                double gpuMs = (t2 - t1) / 1e6;  // Cómputo + Transferencia DMA
                double wrapMs = (t4 - t2) / 1e6; // Creación del objeto Result

                log.debug(">>> GPU DMA BATCH: Total={}ms | Prep={}ms | GPU+DMA={}ms | Wrap={}ms <<<",
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

    // --- CPU Compute Batch (Legacy / Dense Implementation) ---
    // Se mantiene inalterado para compatibilidad y validación cruzada.
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