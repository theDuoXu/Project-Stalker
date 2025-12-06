package projectstalker.benchmark;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import projectstalker.config.RiverConfig;
import projectstalker.config.SimulationConfig;
import projectstalker.config.SimulationConfig.GpuStrategy;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;
import projectstalker.factory.RiverFactory;
import projectstalker.factory.RiverGeometryFactory;
import projectstalker.physics.simulator.ManningBatchProcessor;

import java.util.Arrays;

@Tag("Benchmark")
@Slf4j
public class ManningGpuBenchmark {

    private RiverGeometry geometry;
    private RiverState initialState;
    private SimulationConfig baseConfig;

    // --- CONFIGURACIÓN DE CARGA ---
    // 50k celdas para saturar la GPU y ver el speedup real vs CPU
    private final int CELL_COUNT = 50_000;

    // Umbral de Batch Size para dejar de ejecutar CPU real e interpolar.
    // 50 pasos en CPU para 50k celdas ya tardan bastante (~segundos).
    private final int CPU_EXECUTION_THRESHOLD_BATCH = 50;

    private final float BASE_DISCHARGE = 50.0f; // Caudal base para equilibrio

    @BeforeEach
    void setUp() throws Exception {
        // 1. Geometría Grande
        RiverConfig riverConfig = RiverConfig.builder()
                .totalLength(CELL_COUNT * 50.0f)
                .spatialResolution(50.0f)
                .baseWidth(50.0f)
                .averageSlope(0.001f)
                .baseManning(0.035f)
                .build();

        RiverGeometryFactory factory = new RiverGeometryFactory();
        this.geometry = factory.createRealisticRiver(riverConfig);

        if (this.geometry.getCellCount() != CELL_COUNT) {
            log.warn("El número de celdas generado ({}) difiere del solicitado ({}).",
                    this.geometry.getCellCount(), CELL_COUNT);
        }

        // 2. Configuración Base (Sin estrategia definida aún)
        this.baseConfig = SimulationConfig.builder()
                .cpuProcessorCount(8)
                .useGpuAccelerationOnTransport(false)
                .gpuFullEvolutionStride(1) // Stride 1 para comparar manzanas con manzanas
                .build();

        // 3. WARM-UP HIDRÁULICO INSTANTÁNEO (Analítico)
        log.info("Generando estado inicial estable para {} celdas (Newton-Raphson)...", CELL_COUNT);
        this.initialState = RiverFactory.createSteadyState(this.geometry, BASE_DISCHARGE);

        log.info("Setup Benchmark completado. VRAM Estimada (Batch 5k): ~2 GB.");
    }

    @Test
    @DisplayName("Benchmark: CPU vs GPU Smart (Triangle) vs GPU Full (Rect)")
    void benchmarkMassiveScalability() {
        log.info("=== INICIANDO BENCHMARK MANNING MASIVO (50k Celdas) ===");

        int[] batchSizes = {10, 100, 1_000, 5_000};

        // --- WARM-UP GENERAL ---
        log.info(">> Calentando motores (JIT y Contexto CUDA)...");
        runBatchIteration(10, false, null); // Warmup CPU
        runBatchIteration(100, true, GpuStrategy.SMART_TRUSTED); // Warmup GPU Smart
        runBatchIteration(100, true, GpuStrategy.FULL_EVOLUTION); // Warmup GPU Full
        log.info(">> Calentamiento completado.\n");

        System.out.printf("%-10s | %-15s | %-12s | %-12s | %-10s | %-10s%n",
                "BATCH", "CPU (s)", "SMART (s)", "FULL (s)", "xSMART", "xFULL");
        System.out.println("----------------------------------------------------------------------------------");

        double cpuMsPerStep = 0;

        for (int batchSize : batchSizes) {
            System.gc();

            // 1. CPU (Real o Estimada)
            double cpuTimeMs;
            boolean isCpuEstimated = false;

            if (batchSize > CPU_EXECUTION_THRESHOLD_BATCH && cpuMsPerStep > 0) {
                cpuTimeMs = cpuMsPerStep * batchSize;
                isCpuEstimated = true;
            } else {
                cpuTimeMs = runBatchIteration(batchSize, false, null);
                if (cpuMsPerStep == 0) {
                    cpuMsPerStep = cpuTimeMs / batchSize;
                }
            }

            // 2. GPU SMART (Optimized)
            double gpuSmartTimeMs = runBatchIteration(batchSize, true, GpuStrategy.SMART_TRUSTED);

            // 3. GPU FULL (Robust)
            double gpuFullTimeMs = runBatchIteration(batchSize, true, GpuStrategy.FULL_EVOLUTION);

            // 4. Reporte
            double xSmart = cpuTimeMs / gpuSmartTimeMs;
            double xFull = cpuTimeMs / gpuFullTimeMs;

            String cpuLabel = String.format("%,.2f%s", cpuTimeMs / 1000.0, isCpuEstimated ? "*" : "");

            System.out.printf("%-10d | %-15s | %-12.4f | %-12.4f | %-10.1fx | %-10.1fx%n",
                    batchSize, cpuLabel,
                    gpuSmartTimeMs / 1000.0,
                    gpuFullTimeMs / 1000.0,
                    xSmart, xFull);
        }
        System.out.println("(* = CPU Time Estimated)");
    }

    /**
     * Ejecuta una iteración midiendo tiempo de Cómputo + DMA.
     * Excluye setup de VRAM (en la medida de lo posible, aunque el procesador gestiona el ciclo de vida).
     */
    private double runBatchIteration(int batchSize, boolean useGpu, GpuStrategy strategy) {
        // Generar Hidrograma de Input
        float[] inputs = new float[batchSize];
        Arrays.fill(inputs, 150.0f);

        // Configurar Iteración Específica
        SimulationConfig iterationConfig = baseConfig
                .withUseGpuAccelerationOnManning(useGpu)
                .withGpuStrategy(strategy)
                .withCpuTimeBatchSize(batchSize); // Vital para que el Processor sepa cortar (Full Evolution)

        // Instanciar Procesador
        try (ManningBatchProcessor processor = new ManningBatchProcessor(geometry, iterationConfig)) {

            // WARM-UP INTERNO (Alloc VRAM + Pinned Buffers)
            // Si usamos GPU, ejecutamos una vez "en falso" para que los buffers DMA se reserven.
            // Esto es necesario porque ManningBatchProcessor crea el solver dentro del try-with-resources,
            // pero ManningGpuSolver tiene lazy init.
            if (useGpu) {
                processor.process(inputs, initialState);
            }

            // MEASUREMENT
            long start = System.nanoTime();

            // La llamada 'process' ahora maneja todo internamente
            processor.process(inputs, initialState);

            long end = System.nanoTime();
            return (end - start) / 1_000_000.0;
        }
    }
}