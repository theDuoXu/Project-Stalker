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
    // Ejecutar 50k celdas por 1000 pasos en CPU tardaría demasiado para un test.
    private final int CPU_EXECUTION_THRESHOLD_BATCH = 50;

    private final float BASE_DISCHARGE = 50.0f; // Caudal base para equilibrio

    @BeforeEach
    void setUp() throws Exception {
        // 1. Geometría Grande usando la Factoría
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

        // 3. GENERAR ESTADO INICIAL ESTABLE (Warm-Up Hidráulico)
        // Calculamos el perfil H/V de equilibrio para un caudal constante.
        // Esto es crucial para que la optimización Flyweight de la GPU sea válida.

        log.info("Generando estado inicial estable para {} celdas...", CELL_COUNT);

        float[] qProfile = new float[this.geometry.getCellCount()];
        Arrays.fill(qProfile, BASE_DISCHARGE);

        float[] seedDepth = new float[this.geometry.getCellCount()];
        Arrays.fill(seedDepth, 1.0f);

        // Ejecución síncrona del cálculo de perfil
        ManningProfileCalculatorTask calculator = new ManningProfileCalculatorTask(
                qProfile, seedDepth, this.geometry
        );
        calculator.call();

        // Arrays auxiliares vacíos
        float[] zeros = new float[this.geometry.getCellCount()];

        this.initialState = new RiverState(
                calculator.getCalculatedWaterDepth(), // H equilibrada
                calculator.getCalculatedVelocity(),   // V equilibrada
                zeros, // T
                zeros, // pH
                zeros  // C
        );

        log.info("Setup Benchmark completado. VRAM Estimada (Batch 10k): ~8 GB.");
    }

    @Test
    @DisplayName("Benchmark: Escalabilidad Batch (CPU Interpolada vs GPU Real)")
    void benchmarkMassiveScalability() {
        log.info("=== INICIANDO BENCHMARK MANNING MASIVO (50k Celdas) ===");

        // Tamaños de lote a probar (Variable independiente)
        int[] batchSizes = {10, 100, 1_000, 5_000};

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
     */
    private double runBatchIteration(int batchSize, boolean useGpu) {
        // 1. Preparación de Inputs (Delta Extrinsic)
        float[] newInflows = new float[batchSize];
        Arrays.fill(newInflows, 150.0f); // Caudal de avenida

        // --- Hack de memoria seguro para auxiliares ---
        // Creamos UN SOLO array de ceros del tamaño correcto
        // y lo reutilizamos para todos los pasos de tiempo.
        // Coste de memoria: ~400KB (vs GBs si creamos nuevos arrays por paso)
        int n = this.geometry.getCellCount();
        float[] sharedDummyData = new float[n];

        float[][][] phTmp = new float[batchSize][2][]; // Array de punteros
        for(int k=0; k<batchSize; k++) {
            phTmp[k][0] = sharedDummyData; // Reutilizamos referencia
            phTmp[k][1] = sharedDummyData; // Reutilizamos referencia
        }

        SimulationConfig config = useGpu ? gpuConfig : cpuConfig;

        // 2. Ejecución Controlada
        long start = System.nanoTime();

        // Usamos try-with-resources para asegurar que se libera la VRAM tras cada iteración
        try (ManningBatchProcessor processor = new ManningBatchProcessor(geometry, config)) {
            processor.processBatch(batchSize, initialState, newInflows, phTmp, useGpu);
            // Nota: processBatch ya devuelve un ISimulationResult (Flyweight o Dense),
            // pero para el benchmark solo nos importa el tiempo de retorno.
        }

        long end = System.nanoTime();
        return (end - start) / 1_000_000.0;
    }
}