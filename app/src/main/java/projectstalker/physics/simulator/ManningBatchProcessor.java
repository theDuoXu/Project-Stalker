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
 * <p>
 * Implementa un enfoque híbrido:
 * 1. GPU (Stateful): Usa "Smart Fetch" pasando datos comprimidos a la VRAM.
 * 2. CPU (Concurrent): Usa expansión de matrices en RAM y ejecución multihilo (Legacy/Validation).
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
        // ESTO ARREGLA LOS TESTS UNITARIOS: Evita llamar a JNI si solo queremos testear CPU.
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
     */
    public ManningSimulationResult processBatch(int batchSize,
                                                RiverState initialRiverState,
                                                float[] newInflows,
                                                float[][][] phTmp,
                                                boolean isGpuAccelerated) {
        long startTimeInMilis = System.currentTimeMillis();

        ManningSimulationResult result;

        if (isGpuAccelerated) {
            result = gpuComputeBatch(batchSize, initialRiverState, newInflows, phTmp);
        } else {
            result = cpuComputeBatch(batchSize, initialRiverState, newInflows, phTmp);
        }

        return result.withSimulationTime(System.currentTimeMillis() - startTimeInMilis);
    }

    /**
     * Realiza el cómputo del batch delegando al solver nativo de GPU.
     * Incluye instrumentación de tiempos para perfilado.
     */
    private ManningSimulationResult gpuComputeBatch(int batchSize, RiverState initialRiverState,
                                                    float[] newInflows, float[][][] phTmp) {
        if (this.gpuSolver == null) {
            throw new IllegalStateException("Se solicitó ejecución GPU, pero ManningBatchProcessor fue inicializado en modo solo CPU.");
        }

        try {
            // T0: Inicio preparación
            long t0 = System.nanoTime();

            float[] initialQ = initialRiverState.discharge(this.geometry);
            float[] initialDepths = initialRiverState.waterDepth();

            // T1: Inicio JNI/GPU
            long t1 = System.nanoTime();

            float[][][] results = gpuSolver.solveBatch(initialDepths, newInflows, initialQ);

            // T2: Fin JNI/GPU (Aquí ya tenemos los datos en RAM Java, pero en formato crudo del Solver)
            long t2 = System.nanoTime();

            if (results == null || results.length != batchSize) {
                log.error("Error crítico GPU: BatchSize incorrecto en retorno.");
                return ManningSimulationResult.builder().geometry(this.geometry).states(Collections.emptyList()).build();
            }

            // T3: Inicio Desempaquetado (El cuello de botella de objetos)
            long t3 = System.nanoTime();

            float[][] resultingDepths = new float[batchSize][cellCount];
            float[][] resultingVelocities = new float[batchSize][cellCount];

            // Desempaquetado estructura Solver [Batch][Var][Cell] -> Arrays locales
            for (int i = 0; i < batchSize; i++) {
                resultingDepths[i] = results[i][0];
                resultingVelocities[i] = results[i][1];
            }

            // Creación de objetos RiverState (Costoso)
            List<RiverState> states = assembleRiverStateResults(batchSize, phTmp, resultingDepths, resultingVelocities);

            // T4: Fin total
            long t4 = System.nanoTime();

            // LOG DE PERFILADO (Solo si tarda un poco para no ensuciar logs de tests unitarios rápidos)
            double totalMs = (t4 - t0) / 1e6;
            if (totalMs > 5.0) {
                double prepMs = (t1 - t0) / 1e6;
                double gpuMs = (t2 - t1) / 1e6;   // <--- VELOCIDAD REAL DE TU CÓDIGO C++/CUDA
                double javaMs = (t4 - t3) / 1e6;  // <--- TIEMPO PERDIDO EN OBJETOS JAVA

                log.info(">>> GPU BATCH ({} steps): Total={}ms | Prep={}ms | Native+GPU={}ms | Java Objects={}ms <<<",
                        batchSize,
                        String.format("%.1f", totalMs),
                        String.format("%.1f", prepMs),
                        String.format("%.1f", gpuMs),
                        String.format("%.1f", javaMs)
                );
            }

            return ManningSimulationResult.builder().geometry(this.geometry).states(states).build();

        } catch (Exception e) {
            log.error("Fallo en ejecución GPU", e);
            throw new RuntimeException("Error en simulación GPU", e);
        }
    }

    // --- CPU Compute Batch ---
    private ManningSimulationResult cpuComputeBatch(int batchSize, RiverState initialRiverState,
                                                    float[] newInflows, float[][][] phTmp) {

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
        return ManningSimulationResult.builder().geometry(this.geometry).states(states).build();
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