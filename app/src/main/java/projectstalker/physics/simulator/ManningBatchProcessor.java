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

    // CAMBIO 1: Stateful Solver (Mantiene la sesión C++ activa)
    private final ManningGpuSolver gpuSolver;

    /**
     * Constructor.
     * Inicializa tanto el Pool de CPU como la Sesión de GPU.
     */
    public ManningBatchProcessor(RiverGeometry geometry, SimulationConfig simulationConfig) {
        this.geometry = geometry;
        this.cellCount = geometry.getCellCount();

        // 1. Inicializar CPU Pool (Se mantiene siempre disponible como fallback/validación)
        int processorCount = simulationConfig.getCpuProcessorCount();
        this.threadPool = Executors.newFixedThreadPool(Math.max(processorCount, 1));

        // CAMBIO 2: Inicializar GPU Session (Alloc + Baking)
        // Se pasa la geometría aquí para que C++ haga el 'initSession' una sola vez.
        this.gpuSolver = new ManningGpuSolver(geometry);

        log.info("ManningBatchProcessor inicializado. (CPU Threads: {}, GPU Session Active)", processorCount);
    }

    /**
     * Procesa un batch completo de simulación.
     *
     * @param batchSize         El número de pasos a simular.
     * @param initialRiverState El estado del río al inicio del batch (t=0).
     * @param newInflows        CAMBIO 3: Recibe array 1D [BatchSize] (Raw Input) en lugar de la matriz expandida.
     * @param phTmp             Arrays pre-calculados de [Temperatura, pH].
     * @param isGpuAccelerated  Indica si se debe usar el solver de GPU o CPU.
     */
    public ManningSimulationResult processBatch(int batchSize,
                                                RiverState initialRiverState,
                                                float[] newInflows,
                                                float[][][] phTmp,
                                                boolean isGpuAccelerated) {
        long startTimeInMilis = System.currentTimeMillis();

        ManningSimulationResult result;

        if (isGpuAccelerated) {
            // CAMINO RÁPIDO (GPU): Pasamos datos comprimidos (Smart Fetch)
            result = gpuComputeBatch(batchSize, initialRiverState, newInflows, phTmp);
        } else {
            // CAMINO LEGACY (CPU): Expandimos la matriz en RAM internamente
            result = cpuComputeBatch(batchSize, initialRiverState, newInflows, phTmp);
        }

        return result.withSimulationTime(System.currentTimeMillis() - startTimeInMilis);
    }

    /**
     * Realiza el cómputo del batch delegando al solver nativo de GPU.
     */
    private ManningSimulationResult gpuComputeBatch(int batchSize, RiverState initialRiverState,
                                                    float[] newInflows, float[][][] phTmp) {
        try {
            // 1. Extraer estado inicial necesario para la GPU (t=0)
            float[] initialQ = initialRiverState.discharge(this.geometry);
            float[] initialDepths = initialRiverState.waterDepth();

            // 2. Llamada Stateful (Zero-Copy Pinning en JNI)
            // Devuelve [Batch][Variable][Cell] usando la nueva firma del Solver
            float[][][] results = gpuSolver.solveBatch(initialDepths, newInflows, initialQ);

            // 3. Validación básica
            if (results == null || results.length != batchSize) {
                log.error("Error crítico GPU: BatchSize incorrecto en retorno.");
                return ManningSimulationResult.builder().geometry(this.geometry).states(Collections.emptyList()).build();
            }

            // 4. Re-ensamblar para formato interno
            // GPU devuelve [Step][0=H, 1=V][Cell]. Necesitamos separar H y V para el ensamblador.
            float[][] resultingDepths = new float[batchSize][cellCount];
            float[][] resultingVelocities = new float[batchSize][cellCount];

            for (int i = 0; i < batchSize; i++) {
                resultingDepths[i] = results[i][0];     // Profundidad
                resultingVelocities[i] = results[i][1]; // Velocidad
            }

            List<RiverState> states = assembleRiverStateResults(batchSize, phTmp, resultingDepths, resultingVelocities);
            return ManningSimulationResult.builder().geometry(this.geometry).states(states).build();

        } catch (Exception e) {
            log.error("Fallo en ejecución GPU", e);
            throw new RuntimeException("Error en simulación GPU", e);
        }
    }

    /**
     * Realiza el cómputo del batch utilizando la CPU (Legacy / Integrity Check).
     * Mantiene la lógica original de expansión de matrices y ThreadPool.
     */
    private ManningSimulationResult cpuComputeBatch(int batchSize, RiverState initialRiverState,
                                                    float[] newInflows, float[][][] phTmp) {

        // 1. Expandir matriz de caudales (Lógica original MOVIDA aquí dentro)
        // Solo pagamos este coste si usamos CPU.
        float[][] allDischargeProfiles = createDischargeProfiles(batchSize, newInflows, initialRiverState.discharge(this.geometry));

        List<ManningProfileCalculatorTask> tasks = new ArrayList<>(batchSize);
        float[] initialWaterDepth = initialRiverState.waterDepth();

        // 2. Crear las tareas (Una por paso de tiempo)
        for (int i = 0; i < batchSize; i++) {
            tasks.add(new ManningProfileCalculatorTask(
                    allDischargeProfiles[i], // Usamos la matriz expandida
                    initialWaterDepth,
                    geometry
            ));
        }

        // 3. Ejecutar y esperar (Sin cambios)
        List<Future<ManningProfileCalculatorTask>> futures;
        try {
            futures = threadPool.invokeAll(tasks);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Interrupción en CPU batch.", e);
        }

        // 4. Recoger resultados (Sin cambios)
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

    /**
     * Construye la matriz de perfiles de caudal simulando la propagación de ondas (Lógica CPU Legacy).
     * Se mantiene estrictamente para validación de integridad.
     */
    private float[][] createDischargeProfiles(int batchSize, float[] newDischarges, float[] initialDischarges) {
        float[][] dischargeProfiles = new float[batchSize][cellCount];
        for (int j = 0; j < batchSize; j++) {
            for (int k = 0; k < cellCount; k++) {
                if (k <= j) {
                    // Onda nueva: Viene del array de inputs
                    dischargeProfiles[j][k] = newDischarges[j - k];
                } else {
                    // Onda vieja: Viene del estado inicial del río
                    int sourceIndex = k - (j + 1);
                    if (sourceIndex >= 0) {
                        dischargeProfiles[j][k] = initialDischarges[sourceIndex];
                    } else {
                        // Edge case lógico, fallback seguro
                        dischargeProfiles[j][k] = initialDischarges[0];
                    }
                }
            }
        }
        return dischargeProfiles;
    }

    /**
     * Ensambla los resultados físicos (H, V) con los químicos (T, pH).
     */
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

    /**
     * Cierre limpio de recursos (Threads CPU + Memoria GPU).
     */
    @Override
    public void close() {
        // 1. Cerrar Sesión GPU
        if (gpuSolver != null) {
            gpuSolver.close();
        }
        // 2. Apagar Pool CPU
        if (threadPool != null && !threadPool.isShutdown()) {
            threadPool.shutdown();
        }
        log.info("ManningBatchProcessor cerrado y recursos liberados.");
    }
}