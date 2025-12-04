package projectstalker.benchmark;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import projectstalker.config.RiverConfig;
import projectstalker.config.SimulationConfig;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;
import projectstalker.factory.RiverGeometryFactory;
import projectstalker.physics.impl.ManningProfileCalculatorTask;
import projectstalker.physics.simulator.ManningBatchProcessor;

import java.util.Arrays;

@Tag("Benchmark")
@Slf4j
public class ManningGpuBenchmark {

    private RiverGeometry geometry;
    private RiverState initialState;
    private SimulationConfig cpuConfig;
    private SimulationConfig gpuConfig;

    // --- CONFIGURACIÓN DE CARGA ---
    // 50k celdas para saturar la GPU y ver el speedup real vs CPU
    private final int CELL_COUNT = 50_000;

    // Umbral de Batch Size para dejar de ejecutar CPU real e interpolar.
    private final int CPU_EXECUTION_THRESHOLD_BATCH = 50;

    private final float BASE_DISCHARGE = 50.0f; // Caudal base para equilibrio

    @BeforeEach
    void setUp() throws Exception {
        // 1. Geometría Grande usando la Factoría
        RiverConfig riverConfig = RiverConfig.builder()
                .totalLength(CELL_COUNT * 50.0f)
                .spatialResolution(50.0f)
                .baseWidth(50.0f)
                .averageSlope(0.001f)
                .baseManning(0.035f)
                .build();

        RiverGeometryFactory factory = new RiverGeometryFactory();
        this.geometry = factory.createRealisticRiver(riverConfig);

        // Validación de seguridad
        if (this.geometry.getCellCount() != CELL_COUNT) {
            log.warn("El número de celdas generado ({}) difiere del solicitado ({}). Ajustando lógica...",
                    this.geometry.getCellCount(), CELL_COUNT);
        }

        // 2. Configuraciones de Simulación
        SimulationConfig baseConfig = SimulationConfig.builder()
                .cpuProcessorCount(8)
                .useGpuAccelerationOnTransport(false)
                .build();

        this.cpuConfig = baseConfig.withUseGpuAccelerationOnManning(false);
        this.gpuConfig = baseConfig.withUseGpuAccelerationOnManning(true);

        // 3. GENERAR ESTADO INICIAL ESTABLE (Warm-Up Hidráulico)
        log.info("Generando estado inicial estable para {} celdas...", CELL_COUNT);

        float[] qProfile = new float[this.geometry.getCellCount()];
        Arrays.fill(qProfile, BASE_DISCHARGE);

        float[] seedDepth = new float[this.geometry.getCellCount()];
        Arrays.fill(seedDepth, 1.0f);

        ManningProfileCalculatorTask calculator = new ManningProfileCalculatorTask(
                qProfile, seedDepth, this.geometry
        );
        calculator.call();

        float[] zeros = new float[this.geometry.getCellCount()];

        this.initialState = new RiverState(
                calculator.getCalculatedWaterDepth(),
                calculator.getCalculatedVelocity(),
                zeros, zeros, zeros
        );

        log.info("Setup Benchmark completado. VRAM Estimada (Batch 10k): ~8 GB.");
    }

    @Test
    @DisplayName("Benchmark: Escalabilidad Batch (CPU Interpolada vs GPU Real + DMA)")
    void benchmarkMassiveScalability() {
        log.info("=== INICIANDO BENCHMARK MANNING MASIVO (50k Celdas) ===");

        // Tamaños de lote a probar
        int[] batchSizes = {10, 100, 1_000, 5_000};

        // --- WARM-UP GENERAL (JIT) ---
        log.info(">> Calentando motores (JIT y Contexto CUDA)...");
        runBatchIteration(10, false); // Warmup CPU JIT
        runBatchIteration(100, true); // Warmup GPU Context
        log.info(">> Calentamiento completado.\n");

        System.out.printf("%-15s | %-20s | %-15s | %-15s%n", "BATCH SIZE", "CPU (s)", "GPU (s)", "SPEEDUP");
        System.out.println("----------------------------------------------------------------------------");

        double cpuMsPerStep = 0;

        for (int i = 0; i < batchSizes.length; i++) {
            int batchSize = batchSizes[i];
            System.gc();

            // 1. Lógica CPU
            double cpuTimeMs;
            boolean isCpuEstimated = false;

            if (batchSize > CPU_EXECUTION_THRESHOLD_BATCH && i > 0) {
                cpuTimeMs = cpuMsPerStep * batchSize;
                isCpuEstimated = true;
            } else {
                cpuTimeMs = runBatchIteration(batchSize, false);
                if (cpuMsPerStep == 0) {
                    cpuMsPerStep = cpuTimeMs / batchSize;
                }
            }

            // 2. Medir GPU (Medición Real con DMA optimizado)
            double gpuTimeMs = runBatchIteration(batchSize, true);

            // 3. Reportar
            double speedup = cpuTimeMs / gpuTimeMs;
            double cpuSec = cpuTimeMs / 1000.0;
            double gpuSec = gpuTimeMs / 1000.0;

            String cpuLabel = String.format("%,.2f %s", cpuSec, isCpuEstimated ? "(Est.)" : "");

            System.out.printf("%-15d | %-20s | %-15.4f | %-15.1fx %s%n",
                    batchSize, cpuLabel, gpuSec, speedup,
                    (speedup > 100.0 ? "🚀🚀🚀" : (speedup > 10.0 ? "🚀🚀" : "🚀")));
        }
    }

    /**
     * Ejecuta una iteración de benchmark midiendo el tiempo de procesamiento puro.
     * EXCLUYE: Tiempos de inicialización de sesión, carga de geometría y reservas de memoria iniciales.
     */
    private double runBatchIteration(int batchSize, boolean useGpu) {
        // 1. Preparación de Inputs
        float[] newInflows = new float[batchSize];
        Arrays.fill(newInflows, 150.0f);

        // Hack de memoria seguro para auxiliares
        int n = this.geometry.getCellCount();
        float[] sharedDummyData = new float[n];
        float[][][] phTmp = new float[batchSize][2][];
        for(int k=0; k<batchSize; k++) {
            phTmp[k][0] = sharedDummyData;
            phTmp[k][1] = sharedDummyData;
        }

        SimulationConfig config = useGpu ? gpuConfig : cpuConfig;

        // INSTANCIACIÓN FUERA DEL CRONÓMETRO
        // El benchmark debe medir throughput, no startup time.
        try (ManningBatchProcessor processor = new ManningBatchProcessor(geometry, config)) {

            // --- GPU SPECIFIC WARM-UP ---
            // Si es GPU, forzamos la ejecución de un batch preliminar para:
            // 1. Disparar initSession() -> Cargar geometría y estado a VRAM.
            // 2. Disparar ensureBuffersCapacity() -> Reservar DirectBuffers del tamaño 'batchSize'.
            if (useGpu) {
                processor.processBatch(batchSize, initialState, newInflows, phTmp, true);
            }

            // 2. MEDICIÓN CRÍTICA (Solo Cómputo + DMA)
            long start = System.nanoTime();

            processor.processBatch(batchSize, initialState, newInflows, phTmp, useGpu);

            long end = System.nanoTime();
            return (end - start) / 1_000_000.0;
        }
    }
}