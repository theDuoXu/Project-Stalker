package projectstalker.physics.simulator;

import lombok.extern.slf4j.Slf4j;
import projectstalker.config.SimulationConfig;
import projectstalker.config.SimulationConfig.GpuStrategy;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;
import projectstalker.domain.simulation.IManningResult;
import projectstalker.domain.river.RiverState; // Asegurar import
import projectstalker.factory.SimulationResultFactory;
import projectstalker.physics.impl.ManningProfileCalculatorTask;
import projectstalker.physics.jni.ManningGpuSolver;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

@Slf4j
public class ManningBatchProcessor implements AutoCloseable {

    private final RiverGeometry geometry;
    private final SimulationConfig config;
    private final ExecutorService threadPool;
    private final int cellCount;
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

    // --- MÉTODOS DE FÁBRICA INTERNOS (Protected para Testing) ---
    // Esto permite que el Test sobreescriba este método y devuelva un Mock
    protected ManningGpuSolver createGpuSolver(RiverGeometry geometry) {
        return new ManningGpuSolver(geometry);
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
        // Usamos el método factoría protegido
        try (ManningGpuSolver solver = createGpuSolver(geometry)) {
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
                    rawResult.depths(), rawResult.velocities(), null, rawResult.activeWidth(),
                    System.currentTimeMillis() - startTime
            );
        }
    }

    // --- GPU FULL EVOLUTION ---
    private IManningResult processGpuFullEvolution(float[] fullInflowProfile, RiverState initialState, long startTime) {
        int totalSteps = fullInflowProfile.length;
        int batchSize = config.getCpuTimeBatchSize();
        int stride = config.getGpuFullEvolutionStride();

        // Validación estricta: Los lotes intermedios NO pueden tener padding
        if (batchSize % stride != 0) {
            throw new IllegalArgumentException("Configuración inválida: cpuTimeBatchSize (" + batchSize + ") debe ser múltiplo de gpuFullEvolutionStride (" + stride + ").");
        }

        List<float[]> dChunks = new ArrayList<>();
        List<float[]> vChunks = new ArrayList<>();
        float[] currentQ = initialState.discharge(geometry);
        float[] initialDepths = initialState.waterDepth();

        try (ManningGpuSolver solver = createGpuSolver(geometry)) {
            int processedSteps = 0;

            while (processedSteps < totalSteps) {
                int remainingSteps = totalSteps - processedSteps;
                int logicalBatchSize = Math.min(batchSize, remainingSteps);

                // Cálculo de Padding (Solo aplica al ÚLTIMO lote si logicalBatchSize < batchSize)
                int remainder = logicalBatchSize % stride;
                int paddingNeeded = (remainder == 0) ? 0 : (stride - remainder);
                int effectiveBatchSize = logicalBatchSize + paddingNeeded;

                float[] batchInflows = new float[effectiveBatchSize];
                System.arraycopy(fullInflowProfile, processedSteps, batchInflows, 0, logicalBatchSize);

                if (paddingNeeded > 0) {
                    float lastVal = batchInflows[logicalBatchSize - 1];
                    for (int k = 0; k < paddingNeeded; k++) {
                        batchInflows[logicalBatchSize + k] = lastVal;
                    }
                    log.debug("Batch final extendido con {} pasos de padding.", paddingNeeded);
                }

                ManningGpuSolver.RawGpuResult res = solver.solveFullEvolutionBatch(
                        initialDepths, batchInflows, currentQ, stride
                );

                dChunks.add(res.depths());
                vChunks.add(res.velocities());

                processedSteps += logicalBatchSize;
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

    // --- CPU IMPLEMENTATION (legacy) ---
    private IManningResult processCpu(float[] fullInflowProfile, RiverState initialState, long startTime) {
        int totalSteps = fullInflowProfile.length;

        // 1. Pre-calcular matriz de caudales (Memoria intensiva, comportamiento legacy aceptado)
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
        float[] zeroArr = new float[cellCount]; // Optimización

        for (int i = 0; i < totalSteps; i++) {
            try {
                ManningProfileCalculatorTask task = futures.get(i).get();
                states.add(RiverState.builder()
                        .waterDepth(task.getCalculatedWaterDepth())
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
                config, geometry, finalD, finalV, logicalSteps, time
        );
    }

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
        if (threadPool != null) threadPool.shutdown();
    }
}