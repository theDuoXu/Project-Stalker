package projectstalker.physics.simulator;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import projectstalker.config.RiverConfig;
import projectstalker.config.SimulationConfig;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;
import projectstalker.domain.simulation.ManningSimulationResult;
import projectstalker.factory.RiverGeometryFactory;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

@Tag("GPU")
@Slf4j
class ManningGpuAccuracyTest {

    private ManningBatchProcessor batchProcessor;
    private RiverGeometry realGeometry;

    // Configuraciones explícitas
    private SimulationConfig cpuConfig;
    private SimulationConfig gpuConfig;

    private int cellCount;
    private final int BATCH_SIZE = 5;
    private final double EPSILON = 1e-4; // Tolerancia Float vs Double

    @BeforeEach
    void setUp() {
        // 1. Geometría Real
        RiverConfig riverConfig = RiverConfig.getTestingRiver();
        RiverGeometryFactory factory = new RiverGeometryFactory();
        this.realGeometry = factory.createRealisticRiver(riverConfig);


        this.cellCount = this.realGeometry.getCellCount();
        log.info("Test configurado. Geometría real tiene {} celdas.", this.cellCount);

        // 2. Configuración Base
        SimulationConfig baseConfig = SimulationConfig.builder()
                .riverConfig(riverConfig)
                .seed(12345L)
                .totalTime(3600)
                .deltaTime(10.0f)
                .cpuProcessorCount(4)
                .cpuTimeBatchSize(BATCH_SIZE)
                .useGpuAccelerationOnTransport(false)
                .build();

        // 3. Derivar las dos configuraciones usando .with (Lombok @With)
        this.cpuConfig = baseConfig.withUseGpuAccelerationOnManning(false);
        this.gpuConfig = baseConfig.withUseGpuAccelerationOnManning(true);
    }

    @Test
    @DisplayName("Paridad Numérica: GPU (Float) debe coincidir con CPU (Double) dentro de tolerancia")
    void compareCpuVsGpu_shouldProduceIdenticalResults() {
        log.info("=== INICIANDO TEST DE PRECISIÓN NUMÉRICA ===");

        // --- DATOS DE ENTRADA COMUNES ---
        double[] zeroArray = new double[cellCount];
        double[] initialDepth = new double[cellCount];
        Arrays.fill(initialDepth, 0.5);

        RiverState initialState = new RiverState(
                initialDepth, zeroArray, zeroArray, zeroArray
        );

        double[] flowInput = new double[BATCH_SIZE];
        Arrays.fill(flowInput, 150.0);
        double[] flowInitial = new double[cellCount];
        Arrays.fill(flowInitial, 50.0);

        // Usamos un processor temporal para generar los perfiles (da igual qué config use)
        ManningBatchProcessor helperProcessor = new ManningBatchProcessor(realGeometry, cpuConfig);
        double[][] discharges = helperProcessor.createDischargeProfiles(BATCH_SIZE, flowInput, flowInitial);
        double[][][] phTmp = new double[BATCH_SIZE][2][cellCount]; // Ceros (no afectan hidráulica)

        // --- EJECUCIÓN A: MODO CPU ---
        log.info(">> Ejecutando en CPU...");
        // Instanciamos con cpuConfig para ser consistentes (hilos, etc.)
        ManningBatchProcessor cpuProcessor = new ManningBatchProcessor(realGeometry, cpuConfig);

        long t1 = System.nanoTime();
        ManningSimulationResult resultCpu = cpuProcessor.processBatch(
                BATCH_SIZE, initialState, discharges, phTmp,
                cpuConfig.isUseGpuAccelerationOnManning() // false
        );
        long cpuTime = System.nanoTime() - t1;

        // --- EJECUCIÓN B: MODO GPU ---
        log.info(">> Ejecutando en GPU...");
        // Instanciamos con gpuConfig
        ManningBatchProcessor gpuProcessor = new ManningBatchProcessor(realGeometry, gpuConfig);

        long t2 = System.nanoTime();
        ManningSimulationResult resultGpu = gpuProcessor.processBatch(
                BATCH_SIZE, initialState, discharges, phTmp,
                gpuConfig.isUseGpuAccelerationOnManning() // true
        );
        long gpuTime = System.nanoTime() - t2;

        log.info("Tiempos -> CPU: {}ms | GPU: {}ms | Speedup: {}x",
                cpuTime/1e6, gpuTime/1e6, (double)cpuTime/gpuTime);

        // --- VALIDACIÓN ---
        assertNotNull(resultCpu);
        assertNotNull(resultGpu);

        for (int t = 0; t < BATCH_SIZE; t++) {
            RiverState sCpu = resultCpu.getStates().get(t);
            RiverState sGpu = resultGpu.getStates().get(t);

            compareStates(t, sCpu, sGpu);
        }
        log.info("=== TEST DE PARIDAD SUPERADO CON ÉXITO ===");
    }

    private void compareStates(int step, RiverState cpu, RiverState gpu) {
        for (int i = 0; i < cellCount; i++) {
            // Profundidad
            double hCpu = cpu.getWaterDepthAt(i);
            double hGpu = gpu.getWaterDepthAt(i);
            if (Math.abs(hCpu - hGpu) > EPSILON) {
                fail(String.format("Divergencia H en T=%d C=%d. CPU=%.5f GPU=%.5f", step, i, hCpu, hGpu));
            }

            // Velocidad
            double vCpu = cpu.getVelocityAt(i);
            double vGpu = gpu.getVelocityAt(i);
            if (Math.abs(vCpu - vGpu) > EPSILON) {
                fail(String.format("Divergencia V en T=%d C=%d. CPU=%.5f GPU=%.5f", step, i, vCpu, vGpu));
            }
        }
    }

    @Test
    @DisplayName("Benchmark de Escalabilidad: CPU vs GPU")
    void benchmarkScalability() {
        log.info("=== INICIANDO BENCHMARK DE RENDIMIENTO (CPU vs GPU) ===");

        // Tamaños de lote a probar (ajusta según tu RAM disponible)
        // Nota: 100_000 muestras requieren bastante memoria Heap en Java (-Xmx4G recomendado)
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
            // Forzamos limpieza de memoria antes de cada lote grande
            System.gc();

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
        // Generación de datos FUERA del cronómetro (no queremos medir cuán rápido Java crea arrays)
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
        /*
            Forma muy eficiente de inicializar los datos sin crear arrays temporales extra
            Atención, estamos rompiendo la encapsulación a propósito con Arrays.fill para evitar
            tener que hacer 50000 copias de memoria para un benchmark
         */
        Arrays.fill(initialState.waterDepth(), 0.5); // Rellenar con algo válido

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