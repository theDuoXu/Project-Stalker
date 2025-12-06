package projectstalker.physics.simulator;

import lombok.extern.slf4j.Slf4j;
import projectstalker.config.SimulationConfig;
import projectstalker.config.SimulationConfig.GpuStrategy;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;
import projectstalker.domain.simulation.DenseManningResult;
import projectstalker.domain.simulation.IManningResult;
import projectstalker.factory.SimulationResultFactory;
import projectstalker.physics.impl.ManningProfileCalculatorTask;
import projectstalker.physics.jni.ManningGpuSolver;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Orquestador de la simulación hidráulica.
 * <p>
 * REFACTORIZADO: Ahora gestiona el ciclo de vida completo de la simulación (Chunking/Striding),
 * no solo la ejecución de lotes individuales.
 * <p>
 * Responsabilidades:
 * 1. Decidir la estrategia (CPU vs GPU Smart vs GPU Full Evolution).
 * 2. Gestionar el bucle de tiempo y la paginación de memoria.
 * 3. Utilizar {@link SimulationResultFactory} para producir el resultado optimizado correcto.
 */
@Slf4j
public class ManningBatchProcessor implements AutoCloseable {

    private final RiverGeometry geometry;
    private final SimulationConfig config;
    private final ExecutorService threadPool;
    private final int cellCount;

    // Solver GPU (Lazy initialization)
    // Se instancia solo si la config lo pide.
    private final boolean useGpu;

    public ManningBatchProcessor(RiverGeometry geometry, SimulationConfig config) {
        this.geometry = geometry;
        this.config = config;
        this.cellCount = geometry.getCellCount();
        this.useGpu = config.isUseGpuAccelerationOnManning();
        int processorCount = config.getCpuProcessorCount();
        this.threadPool = Executors.newFixedThreadPool(Math.max(processorCount, 1));
        log.info("ManningBatchProcessor inicializado. (GPU: {}, Stride: {})", useGpu, config.getGpuFullEvolutionStride());
    }

    public IManningResult process(float[] fullInflowProfile, RiverState initialState) {
        long startTime = System.currentTimeMillis();
        if (useGpu) {
            GpuStrategy strategy = config.getGpuStrategy();
            if (strategy == GpuStrategy.SMART_SAFE || strategy == GpuStrategy.SMART_TRUSTED) {
                return processGpuSmart(fullInflowProfile, initialState, startTime, strategy);
            } else {
                return processGpuFullEvolution(fullInflowProfile, initialState, startTime);
            }
        } else {
            return processCpu(fullInflowProfile, initialState, startTime);
        }
    }

    // --- GPU SMART ---
    private IManningResult processGpuSmart(float[] inflowProfile, RiverState initialState, long startTime, GpuStrategy strategy) {
        try (ManningGpuSolver solver = new ManningGpuSolver(geometry)) {
            ManningGpuSolver.RawGpuResult rawResult;
            try {
                rawResult = solver.solveSmartBatch(
                        initialState.waterDepth(),
                        inflowProfile,
                        initialState.discharge(geometry),
                        strategy == GpuStrategy.SMART_TRUSTED
                );
            } catch (IllegalStateException e) {
                log.warn("Fallback Smart -> Full Evolution: {}", e.getMessage());
                return processGpuFullEvolution(inflowProfile, initialState, startTime);
            }
            return SimulationResultFactory.createSmartGpuResult(
                    config, geometry, initialState,
                    rawResult.depths(), rawResult.velocities(), null,
                    System.currentTimeMillis() - startTime
            );
        }
    }

    // --- GPU FULL EVOLUTION (Con Padding) ---
    private IManningResult processGpuFullEvolution(float[] fullInflowProfile, RiverState initialState, long startTime) {
        int totalSteps = fullInflowProfile.length;
        int batchSize = config.getCpuTimeBatchSize();
        int stride = config.getGpuFullEvolutionStride();

        // Validación inicial de config
        if (batchSize % stride != 0) {
            throw new IllegalArgumentException("Configuración inválida: cpuTimeBatchSize debe ser múltiplo de gpuFullEvolutionStride.");
        }

        List<float[]> dChunks = new ArrayList<>();
        List<float[]> vChunks = new ArrayList<>();
        float[] currentQ = initialState.discharge(geometry);
        float[] initialDepths = initialState.waterDepth();

        try (ManningGpuSolver solver = new ManningGpuSolver(geometry)) {
            int processedSteps = 0;

            while (processedSteps < totalSteps) {
                // 1. Determinar tamaño lógico restante
                int remainingSteps = totalSteps - processedSteps;
                int logicalBatchSize = Math.min(batchSize, remainingSteps);

                // 2. Cálculo de Padding (Relleno)
                // Si el lote no llena el stride, extendemos el input para evitar perder datos.
                // Esto garantiza que el kernel siempre reciba múltiplo de stride.
                int remainder = logicalBatchSize % stride;
                int paddingNeeded = (remainder == 0) ? 0 : (stride - remainder);
                int effectiveBatchSize = logicalBatchSize + paddingNeeded;

                // 3. Construcción del Input Array (Con Padding si es necesario)
                float[] batchInflows = new float[effectiveBatchSize];

                // Copiar datos reales
                System.arraycopy(fullInflowProfile, processedSteps, batchInflows, 0, logicalBatchSize);

                // Rellenar padding con el último valor válido (proyección constante)
                if (paddingNeeded > 0) {
                    float lastVal = batchInflows[logicalBatchSize - 1];
                    for (int k = 0; k < paddingNeeded; k++) {
                        batchInflows[logicalBatchSize + k] = lastVal;
                    }
                    log.debug("Batch final extendido con {} pasos de padding para alineación de stride.", paddingNeeded);
                }

                // 4. Ejecución
                ManningGpuSolver.RawGpuResult res = solver.solveFullEvolutionBatch(
                        initialDepths, batchInflows, currentQ, stride
                );

                dChunks.add(res.depths());
                vChunks.add(res.velocities());

                processedSteps += logicalBatchSize; // Solo avanzamos lo que realmente procesamos del input original
            }

            long execTime = System.currentTimeMillis() - startTime;
            long totalElements = dChunks.stream().mapToLong(a -> a.length).sum();

            if (totalElements < (Integer.MAX_VALUE - 100)) {
                return flattenChunksToStrided(dChunks, vChunks, totalSteps, execTime);
            } else {
                return SimulationResultFactory.createChunkedGpuResult(
                        config, geometry, dChunks, vChunks,
                        (batchSize / stride), totalSteps, execTime
                );
            }
        }
    }

    // --- ESTRATEGIA 3: CPU (Legacy / Reference) ---

    private IManningResult processCpu(float[] fullInflowProfile, RiverState initialState, long startTime) {
        // En modo CPU, procesamos todo de una vez (o podríamos chunkear también, pero mantenemos lógica original)
        // La implementación original usaba 'processBatch' para un bloque. Aquí adaptamos para procesar todo.
        // Si el perfil es muy grande, esto podría dar OOM, pero es el comportamiento esperado "Legacy".

        int totalSteps = fullInflowProfile.length;
        float[][] allDischargeProfiles = createDischargeProfiles(totalSteps, fullInflowProfile, initialState.discharge(geometry));
        float[] initialWaterDepth = initialState.waterDepth();

        List<ManningProfileCalculatorTask> tasks = new ArrayList<>(totalSteps);
        for (int i = 0; i < totalSteps; i++) {
            tasks.add(new ManningProfileCalculatorTask(allDischargeProfiles[i], initialWaterDepth, geometry));
        }

        List<Future<ManningProfileCalculatorTask>> futures;
        try {
            futures = threadPool.invokeAll(tasks);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Simulación CPU interrumpida.", e);
        }

        List<RiverState> states = new ArrayList<>(totalSteps);
        float[] zeroArr = new float[cellCount]; // Optimización memoria para vars no calculadas

        for (int i = 0; i < totalSteps; i++) {
            try {
                ManningProfileCalculatorTask task = futures.get(i).get();
                // Construimos RiverState inmediatamente
                states.add(RiverState.builder()
                        .waterDepth(task.getCalculatedWaterDepth()) // El task devuelve un nuevo array
                        .velocity(task.getCalculatedVelocity())
                        .temperature(zeroArr)
                        .ph(zeroArr)
                        .contaminantConcentration(zeroArr)
                        .build());
            } catch (Exception e) {
                throw new RuntimeException("Error en cálculo CPU paso " + i, e);
            }
        }

        return SimulationResultFactory.createCpuResult(
                geometry,
                states,
                System.currentTimeMillis() - startTime
        );
    }

    // --- HELPERS ---

    private IManningResult flattenChunksToStrided(List<float[]> dChunks, List<float[]> vChunks, int logicalSteps, long time) {
        int totalSize = dChunks.stream().mapToInt(a -> a.length).sum();
        float[] finalD = new float[totalSize];
        float[] finalV = new float[totalSize];

        int ptr = 0;
        for (int i = 0; i < dChunks.size(); i++) {
            float[] d = dChunks.get(i);
            float[] v = vChunks.get(i);
            System.arraycopy(d, 0, finalD, ptr, d.length);
            System.arraycopy(v, 0, finalV, ptr, v.length);
            ptr += d.length;
        }

        return SimulationResultFactory.createStridedGpuResult(
                config, geometry,
                finalD, finalV,
                logicalSteps,
                time
        );
    }

    // Helper lógica CPU original
    private float[][] createDischargeProfiles(int batchSize, float[] newDischarges, float[] initialDischarges) {
        float[][] dischargeProfiles = new float[batchSize][cellCount];
        for (int j = 0; j < batchSize; j++) {
            for (int k = 0; k < cellCount; k++) {
                if (k <= j) {
                    dischargeProfiles[j][k] = newDischarges[j - k];
                } else {
                    int sourceIndex = k - (j + 1);
                    dischargeProfiles[j][k] = (sourceIndex >= 0) ? initialDischarges[sourceIndex] : initialDischarges[0];
                }
            }
        }
        return dischargeProfiles;
    }

    @Override
    public void close() {
        if (threadPool != null && !threadPool.isShutdown()) {
            threadPool.shutdown();
        }
        log.info("ManningBatchProcessor cerrado.");
    }
}