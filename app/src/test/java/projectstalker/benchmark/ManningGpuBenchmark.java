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
    // 100k celdas para saturar la GPU y ver el speedup real vs CPU
    private final int CELL_COUNT = 100_000;

    // Umbral de Batch Size para dejar de ejecutar CPU real e interpolar.
    // Ejecutar 100k celdas por 1000 pasos en CPU tardaría demasiado para un test.
    private final int CPU_EXECUTION_THRESHOLD_BATCH = 50;

    @BeforeEach
    void setUp() {
        // 1. Geometría Grande usando la Factoría (Inspirado en TransportGpuBenchmark)
        RiverConfig riverConfig = RiverConfig.builder()
                .totalLength(CELL_COUNT * 50.0f) // Longitud total para obtener el número de celdas deseado
                .spatialResolution(50.0f)
                .baseWidth(50.0f)
                .averageSlope(0.001f)
                .baseManning(0.035f) // Manning estándar
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
                .cpuProcessorCount(8) // Asumimos máquina potente para competencia justa
                .useGpuAccelerationOnTransport(false)
                .build();

        this.cpuConfig = baseConfig.withUseGpuAccelerationOnManning(false);
        this.gpuConfig = baseConfig.withUseGpuAccelerationOnManning(true);

        // 3. Estado Inicial
        float[] h = new float[geometry.getCellCount()]; Arrays.fill(h, 1.0f); // 1m profundidad
        float[] v = new float[geometry.getCellCount()]; Arrays.fill(v, 1.0f); // 1m/s velocidad
        float[] zeros = new float[geometry.getCellCount()];

        // Creamos estado inicial completo para evitar nulos
        this.initialState = RiverState.builder()
                .waterDepth(h)
                .velocity(v)
                .temperature(zeros)
                .ph(zeros)
                .contaminantConcentration(zeros)
                .build();

        log.info("Setup Benchmark: {} celdas. VRAM Estimada (Batch 10k): ~8 GB.",
                String.format("%,d", geometry.getCellCount()));
    }

    @Test
    @DisplayName("Benchmark: Escalabilidad Batch (CPU Interpolada vs GPU Real)")
    void benchmarkMassiveScalability() {
        log.info("=== INICIANDO BENCHMARK MANNING MASIVO (100k Celdas) ===");

        // Tamaños de lote a probar (Variable independiente)
        // Nota: 20k batches * 100k celdas * 8 bytes output = ~16 GB VRAM. Seguro en 3090/4090.
        int[] batchSizes = {10, 100, 1_000, 5_000, 10_000};

        // --- WARM-UP ---
        log.info(">> Calentando motores (JIT y Contexto CUDA)...");
        runBatchIteration(10, false); // Warmup CPU
        runBatchIteration(100, true); // Warmup GPU
        log.info(">> Calentamiento completado.\n");

        System.out.printf("%-15s | %-20s | %-15s | %-15s%n", "BATCH SIZE", "CPU (s)", "GPU (s)", "SPEEDUP");
        System.out.println("----------------------------------------------------------------------------");

        // Variables para interpolación CPU
        double cpuMsPerStep = 0;

        for (int i = 0; i < batchSizes.length; i++) {
            int batchSize = batchSizes[i];
            System.gc(); // Limpieza antes de alocaciones grandes

            // 1. Lógica CPU: Ejecutar o Estimar
            double cpuTimeMs;
            boolean isCpuEstimated = false;

            if (batchSize > CPU_EXECUTION_THRESHOLD_BATCH && i > 0) {
                // Interpolación lineal: T = (ms/step) * batchSize
                cpuTimeMs = cpuMsPerStep * batchSize;
                isCpuEstimated = true;
            } else {
                // Ejecución Real (Solo para batches pequeños para sacar la media)
                cpuTimeMs = runBatchIteration(batchSize, false);

                // Calculamos la métrica base si es una ejecución válida
                if (cpuMsPerStep == 0) {
                    cpuMsPerStep = cpuTimeMs / batchSize;
                    log.info("   [Calibración CPU] Velocidad medida: {} ms/step", String.format("%.3f", cpuMsPerStep));
                }
            }

            // 2. Medir GPU (Siempre Real)
            // La GPU debe aguantar el batch masivo gracias a la memoria adaptativa
            double gpuTimeMs = runBatchIteration(batchSize, true);

            // 3. Reportar
            double speedup = cpuTimeMs / gpuTimeMs;

            // Conversión a segundos para legibilidad
            double cpuSec = cpuTimeMs / 1000.0;
            double gpuSec = gpuTimeMs / 1000.0;

            String cpuLabel = String.format("%,.2f %s", cpuSec, isCpuEstimated ? "(Est.)" : "");

            System.out.printf("%-15d | %-20s | %-15.4f | %-15.1fx %s%n",
                    batchSize, cpuLabel, gpuSec, speedup,
                    (speedup > 100.0 ? "🚀🚀🚀" : (speedup > 10.0 ? "🚀🚀" : "🚀")));
        }
    }

    /**
     * Ejecuta una iteración de benchmark midiendo el tiempo de procesamiento.
     * Utiliza la nueva API Stateful/Smart Fetch.
     */
    private double runBatchIteration(int batchSize, boolean useGpu) {
        // 1. Preparación de Inputs (Chorizo 1D comprimido)
        float[] newInflows = new float[batchSize];
        Arrays.fill(newInflows, 150.0f);

        // Hack de memoria para phTmp: Usamos arrays vacíos compartidos
        // para no desperdiciar Heap Java en datos que no afectan a la hidráulica
        float[][][] phTmp = new float[batchSize][2][0];
        float[] dummyArray = new float[0];
        for(int k=0; k<batchSize; k++) {
            phTmp[k][0] = dummyArray;
            phTmp[k][1] = dummyArray;
        }

        SimulationConfig config = useGpu ? gpuConfig : cpuConfig;

        // 2. Ejecución Controlada (Try-with-resources para liberar GPU)
        long start = System.nanoTime();

        // Incluimos la creación del Processor en el tiempo para ser honestos con el overhead de init
        try (ManningBatchProcessor processor = new ManningBatchProcessor(geometry, config)) {

            // En caso GPU: Medimos Init + Kernel + Transferencias
            // En caso CPU: Medimos Expansión Matriz + Cálculo Concurrent
            processor.processBatch(batchSize, initialState, newInflows, phTmp, useGpu);

            long end = System.nanoTime();
            return (end - start) / 1_000_000.0;
        }
        // Al salir, processor.close() -> native.destroySession() libera VRAM
    }
}