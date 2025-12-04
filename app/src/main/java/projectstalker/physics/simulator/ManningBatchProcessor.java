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
 * 1. GPU (Stateful): Usa "Smart Fetch" y devuelve un {@link FlyweightManningResult} virtualizado.
 * 2. CPU (Concurrent): Usa expansión de matrices en RAM y devuelve un {@link DenseManningResult} materializado.
 * <p>
 * Implementa AutoCloseable para garantizar la liberación de recursos (Threads y GPU Session).
 */
@Slf4j
public class ManningBatchProcessor implements AutoCloseable {

    private final RiverGeometry geometry;
    private final ExecutorService threadPool;
    private final int cellCount;

    // Stateful Solver: Puede ser null si la configuración deshabilita la GPU
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

        // 1. Inicializar CPU Pool (Siempre disponible)
        int processorCount = simulationConfig.getCpuProcessorCount();
        this.threadPool = Executors.newFixedThreadPool(Math.max(processorCount, 1));

        // 2. Inicializar GPU Session (Solo si está habilitado en config)
        // Evita llamar a JNI si solo queremos testear CPU.
        if (simulationConfig.isUseGpuAccelerationOnManning()) {
            log.info("Inicializando sesión GPU...");
            this.gpuSolver = new ManningGpuSolver(geometry);
        } else {
            log.info("Modo solo CPU: No se inicializará la sesión GPU.");
            this.gpuSolver = null;
        }

        log.info("ManningBatchProcessor inicializado. (CPU Threads: {})", processorCount);
    }

    /**
     * Procesa un batch completo de simulación.
     * Devuelve una interfaz común para desacoplar el almacenamiento de datos.
     */
    public ISimulationResult processBatch(int batchSize,
                                          RiverState initialRiverState,
                                          float[] newInflows,
                                          float[][][] phTmp,
                                          boolean isGpuAccelerated) {
        long startTimeInMilis = System.currentTimeMillis();

        ISimulationResult result;

        if (isGpuAccelerated) {
            result = gpuComputeBatch(batchSize, initialRiverState, newInflows, phTmp);
        } else {
            result = cpuComputeBatch(batchSize, initialRiverState, newInflows, phTmp);
        }

        return result; // El tiempo de simulación se encapsula dentro de la implementación específica si es necesario
    }

    /**
     * Realiza el cómputo del batch delegando al solver nativo de GPU.
     * Devuelve un resultado ligero (Flyweight) para evitar saturar la RAM de Java.
     */
    private ISimulationResult gpuComputeBatch(int batchSize, RiverState initialRiverState,
                                              float[] newInflows, float[][][] phTmp) {
        if (this.gpuSolver == null) {
            throw new IllegalStateException("Se solicitó ejecución GPU, pero ManningBatchProcessor fue inicializado en modo solo CPU.");
        }

        try {
            // T0: Inicio preparación
            long t0 = System.nanoTime();

            // Extraemos arrays primitivos para pasar al JNI (Zero-copy friendly)
            float[] initialQ = initialRiverState.discharge(this.geometry);
            float[] initialDepths = initialRiverState.waterDepth();

            // T1: Inicio JNI/GPU
            long t1 = System.nanoTime();

            // EJECUCIÓN NATIVA
            // Devuelve la matriz compacta [Batch][Var][ActiveWidth]
            float[][][] packedResults = gpuSolver.solveBatch(initialDepths, newInflows, initialQ);

            // T2: Fin JNI/GPU
            long t2 = System.nanoTime();

            if (packedResults == null || packedResults.length != batchSize) {
                log.error("Error crítico GPU: BatchSize incorrecto en retorno.");
                // Retorno seguro vacío usando implementación densa
                return DenseManningResult.builder()
                        .geometry(this.geometry)
                        .states(Collections.emptyList())
                        .simulationTime(0)
                        .build();
            }

            // T3: Construcción del Flyweight (Virtualización)
            // Ya NO desempaquetamos los arrays en un bucle. Pasamos los datos crudos al wrapper.
            // Esto reduce el tiempo de "Java Objects" de ms a ns.

            long simulationTimeMs = (t2 - t0) / 1_000_000; // Aproximado para el record

            ISimulationResult result = new FlyweightManningResult(
                    this.geometry,
                    simulationTimeMs,
                    initialRiverState, // Intrinsic State (Base)
                    packedResults,     // Extrinsic State (Delta GPU)
                    phTmp              // Aux Data
            );

            // T4: Fin total
            long t4 = System.nanoTime();

            // LOG DE PERFILADO
            double totalMs = (t4 - t0) / 1e6;
            if (totalMs > 5.0) {
                double prepMs = (t1 - t0) / 1e6;
                double gpuMs = (t2 - t1) / 1e6;
                double javaMs = (t4 - t2) / 1e6;

                log.info(">>> GPU BATCH (Flyweight): Total={}ms | Prep={}ms | Native={}ms | Wrapper={}ms <<<",
                        String.format("%.1f", totalMs),
                        String.format("%.1f", prepMs),
                        String.format("%.1f", gpuMs),
                        String.format("%.1f", javaMs)
                );
            }

            return result;

        } catch (Exception e) {
            log.error("Fallo en ejecución GPU", e);
            throw new RuntimeException("Error en simulación GPU", e);
        }
    }

    // --- CPU Compute Batch (Legacy / Dense Implementation) ---
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

        // En CPU seguimos materializando la lista completa (Dense)
        List<RiverState> states = assembleRiverStateResults(batchSize, phTmp, resultingDepths, resultingVelocities);

        long tEnd = System.currentTimeMillis();

        return DenseManningResult.builder()
                .geometry(this.geometry)
                .states(states)
                .simulationTime(tEnd - tStart)
                .build();
    }

    // Método Legacy privado (Solo para CPU)
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