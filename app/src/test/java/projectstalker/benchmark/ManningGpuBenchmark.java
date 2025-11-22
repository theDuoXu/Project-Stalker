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

    private RiverGeometry realGeometry;
    private SimulationConfig cpuConfig;
    private SimulationConfig gpuConfig;
    private int cellCount;

    @BeforeEach
    void setUp() {
        RiverConfig riverConfig = RiverConfig.getTestingRiver();
        RiverGeometryFactory factory = new RiverGeometryFactory();
        this.realGeometry = factory.createRealisticRiver(riverConfig);
        this.cellCount = this.realGeometry.getCellCount();

        SimulationConfig baseConfig = SimulationConfig.builder()
                .riverConfig(riverConfig)
                .seed(12345L)
                .totalTime(3600)
                .deltaTime(10.0f)
                .cpuProcessorCount(4)
                .useGpuAccelerationOnTransport(false)
                // Batch size se sobrescribe en el benchmark dinámicamente si es necesario,
                // pero aquí ponemos un valor por defecto.
                .cpuTimeBatchSize(100)
                .build();

        this.cpuConfig = baseConfig.withUseGpuAccelerationOnManning(false);
        this.gpuConfig = baseConfig.withUseGpuAccelerationOnManning(true);
    }

    @Test
    @DisplayName("Benchmark de Escalabilidad: CPU vs GPU")
    void benchmarkScalability() {
        log.info("=== INICIANDO BENCHMARK DE RENDIMIENTO (CPU vs GPU) ===");
        log.info("Geometría del río: {} celdas.", this.cellCount);

        // Tamaños de lote a probar
        int[] batchSizes = {100, 1_000, 10_000, 50_000};

        // --- FASE 0: CALENTAMIENTO (WARM-UP) ---
        log.info(">> Calentando motores (JIT y Contexto CUDA)...");
        runBenchmarkIteration(100, false); // Warmup CPU
        runBenchmarkIteration(100, true);  // Warmup GPU
        log.info(">> Calentamiento completado.\n");

        System.out.printf("%-15s | %-15s | %-15s | %-15s%n", "BATCH SIZE", "CPU (ms)", "GPU (ms)", "SPEEDUP");
        System.out.println("---------------------------------------------------------------------");

        // --- FASE 1: BUCLE DE PRUEBA ---
        for (int size : batchSizes) {
            System.gc(); // Limpieza de memoria

            // 1. Ejecutar CPU
            double cpuTimeMs = runBenchmarkIteration(size, false);

            // 2. Ejecutar GPU
            double gpuTimeMs = runBenchmarkIteration(size, true);

            // 3. Calcular Speedup
            double speedup = cpuTimeMs / gpuTimeMs;

            // 4. Reportar tabla
            System.out.printf("%-15d | %-15.2f | %-15.2f | %-15.2fx %s%n",
                    size, cpuTimeMs, gpuTimeMs, speedup,
                    (speedup > 1.0 ? "🚀" : "🐌"));
        }
    }

    /**
     * Ejecuta una iteración aislada y devuelve el tiempo en milisegundos.
     */
    private double runBenchmarkIteration(int batchSize, boolean useGpu) {
        // Generación de datos FUERA del cronómetro
        double[] flowInput = new double[batchSize];
        Arrays.fill(flowInput, 150.0);
        double[] flowInitial = new double[cellCount];
        Arrays.fill(flowInitial, 50.0);

        // Usamos un processor temporal para preparar inputs
        ManningBatchProcessor setupProc = new ManningBatchProcessor(realGeometry, cpuConfig);
        double[][] discharges = setupProc.createDischargeProfiles(batchSize, flowInput, flowInitial);
        double[][][] phTmp = new double[batchSize][2][cellCount];

        RiverState initialState = new RiverState(
                new double[cellCount], new double[cellCount],
                new double[cellCount], new double[cellCount]
        );

        // Inicialización rápida para el benchmark (rompiendo encapsulación para velocidad de setup)
        Arrays.fill(initialState.waterDepth(), 0.5);

        // Selección de configuración
        SimulationConfig configToUse = useGpu ? gpuConfig : cpuConfig;
        ManningBatchProcessor processor = new ManningBatchProcessor(realGeometry, configToUse);

        // --- MEDICIÓN CRÍTICA ---
        long start = System.nanoTime();

        processor.processBatch(
                batchSize, initialState, discharges, phTmp,
                configToUse.isUseGpuAccelerationOnManning()
        );

        long end = System.nanoTime();
        // -----------------------

        return (end - start) / 1_000_000.0;
    }
}