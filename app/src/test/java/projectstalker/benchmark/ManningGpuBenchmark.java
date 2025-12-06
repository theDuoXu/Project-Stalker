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
                .build();

        // 3. WARM-UP HIDRÁULICO INSTANTÁNEO (Analítico)
        log.info("Generando estado inicial estable para {} celdas (Newton-Raphson)...", CELL_COUNT);
        this.initialState = RiverFactory.createSteadyState(this.geometry, BASE_DISCHARGE);

        log.info("Setup Benchmark completado. VRAM Estimada (Batch 5k): ~2 GB.");
    }

    @Test
    @DisplayName("Benchmark A: CPU vs GPU (Stride=1) - High Fidelity")
    void benchmarkMassiveScalability() {
        runBenchmarkSuite(1, "STRIDE 1 (High Fidelity)");
    }

    @Test
    @DisplayName("Benchmark B: CPU vs GPU (Stride=100) - Long Run / Visualization")
    void benchmarkMassiveScalabilityWithStride() {
        runBenchmarkSuite(100, "STRIDE 100 (Long Run)");
    }

    /**
     * Motor genérico del Benchmark.
     */
    private void runBenchmarkSuite(int stride, String label) {
        log.info("=== INICIANDO BENCHMARK MANNING MASIVO (50k Celdas) - {} ===", label);

        // Ajustamos los tamaños de batch para que sean múltiplos del stride (requisito estricto)
        // Ejemplo: Si stride=100, usamos 100, 1000, 5000.
        // Si stride=1, usamos 10, 100, 1000, 5000.
        int[] batchSizes;
        if (stride == 1) {
            batchSizes = new int[]{10, 100, 1_000, 5_000};
        } else {
            batchSizes = new int[]{100, 1_000, 5_000, 10_000}; // Batches más grandes para notar el ahorro
        }

        // --- WARM-UP GENERAL ---
        log.info(">> Calentando motores (JIT y Contexto CUDA)...");
        // Warmup con stride 1 siempre es seguro para Smart
        runBatchIteration(100, true, GpuStrategy.SMART_TRUSTED, 1);
        // Warmup Full con el stride objetivo
        int warmupBatch = Math.max(100, stride);
        runBatchIteration(warmupBatch, true, GpuStrategy.FULL_EVOLUTION, stride);
        log.info(">> Calentamiento completado.\n");

        System.out.printf("%-10s | %-15s | %-12s | %-12s | %-10s | %-10s%n",
                "BATCH", "CPU (s)", "SMART (s)", "FULL (s)", "xSMART", "xFULL");
        System.out.println("----------------------------------------------------------------------------------");

        double cpuMsPerStep = 0;

        for (int batchSize : batchSizes) {
            System.gc();

            // 1. CPU (Real o Estimada)
            // CPU ignora el stride en el cálculo físico, pero se beneficia al guardar menos resultados.
            // Para simplificar la comparativa de "Physics Time", estimamos linealmente.
            double cpuTimeMs;
            boolean isCpuEstimated = false;

            if (batchSize > CPU_EXECUTION_THRESHOLD_BATCH && cpuMsPerStep > 0) {
                cpuTimeMs = cpuMsPerStep * batchSize;
                isCpuEstimated = true;
            } else {
                // CPU Reference siempre Stride 1 para medir coste físico base
                cpuTimeMs = runBatchIteration(batchSize, false, null, 1);
                if (cpuMsPerStep == 0) {
                    cpuMsPerStep = cpuTimeMs / batchSize;
                }
            }

            // 2. GPU SMART (Optimized)
            // Smart SIEMPRE usa Stride=1 internamente (Triángulo Denso).
            double gpuSmartTimeMs = runBatchIteration(batchSize, true, GpuStrategy.SMART_TRUSTED, 1);

            // 3. GPU FULL (Robust)
            // Full Evolution SÍ usa el Stride para optimizar PCIe y Memoria.
            double gpuFullTimeMs = runBatchIteration(batchSize, true, GpuStrategy.FULL_EVOLUTION, stride);

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
    private double runBatchIteration(int totalStepsToSimulate, boolean useGpu, GpuStrategy strategy, int stride) {
        // Generar Hidrograma de Input (Simulación Larga)
        float[] inputs = new float[totalStepsToSimulate];
        Arrays.fill(inputs, 150.0f);

        // CALCULAR BATCH SIZE SEGURO (Límite de Buffer Java ~2GB)
        // Si usamos stride, el buffer de salida es menor, así que podemos procesar lotes más grandes.
        // Stride 1: 2000 pasos. Stride 100: 200,000 pasos (teórico).
        // Mantenemos 2000 como base conservadora pero multiplicamos por stride (hasta un límite razonable).
        int safeBase = 2000;
        int maxSafeBatch = Math.min(safeBase * stride, 10_000); // Tope 10k para no saturar VRAM de inputs

        // Ajustamos para que sea múltiplo del stride (Requisito JNI)
        int safeBatchSize = (maxSafeBatch / stride) * stride;
        if (safeBatchSize == 0) safeBatchSize = stride;

        // Configurar Iteración
        SimulationConfig iterationConfig = baseConfig
                .withUseGpuAccelerationOnManning(useGpu)
                .withGpuStrategy(strategy)
                .withCpuTimeBatchSize(safeBatchSize) // Tamaño de "bocado" seguro
                .withGpuFullEvolutionStride(stride); // Factor de compresión

        // Instanciar Procesador
        try (ManningBatchProcessor processor = new ManningBatchProcessor(geometry, iterationConfig)) {

            // WARM-UP (Solo la primera vez para alojar buffers)
            if (useGpu) {
                float[] warmupInput = new float[Math.min(stride * 2, totalStepsToSimulate)];
                processor.process(warmupInput, initialState);
            }

            // MEASUREMENT
            long start = System.nanoTime();

            // El procesador gestiona el chunking si totalSteps > safeBatchSize
            processor.process(inputs, initialState);

            long end = System.nanoTime();
            return (end - start) / 1_000_000.0;
        }
    }
}