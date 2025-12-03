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

    private RiverGeometry realGeometry;

    // Configuraciones explícitas
    private SimulationConfig cpuConfig;
    private SimulationConfig gpuConfig;

    private int cellCount;
    private final int BATCH_SIZE = 5;
    // Tolerancia: Float (GPU) vs Double (CPU) pueden divergir ligeramente por precisión
    private final float EPSILON = 1e-4f;

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
                .useGpuAccelerationOnTransport(false)
                .build();

        // 3. Derivar las dos configuraciones
        this.cpuConfig = baseConfig.withUseGpuAccelerationOnManning(false);
        this.gpuConfig = baseConfig.withUseGpuAccelerationOnManning(true);
    }

    @Test
    @DisplayName("Paridad Numérica: GPU (Float) debe coincidir con CPU (Double) dentro de tolerancia")
    void compareCpuVsGpu_shouldProduceIdenticalResults() {
        log.info("=== INICIANDO TEST DE PRECISIÓN NUMÉRICA ===");

        // --- DATOS DE ENTRADA COMUNES ---

        // 1. Estado Inicial (t=0)
        float[] initialDepth = new float[cellCount];
        Arrays.fill(initialDepth, 0.5f);
        float[] initialVelocity = new float[cellCount];
        Arrays.fill(initialVelocity, 1.0f); // Velocidad arbitraria para generar caudal
        float[] zeroArray = new float[cellCount];

        // Importante: El estado inicial debe ser consistente porque la GPU lee de él (Smart Fetch Región 2)
        // Calculamos el caudal inicial aproximado (Q = V * A aprox) o simplemente llenamos con valor conocido.
        // Aquí usamos el método de RiverState para consistencia si es posible, o inyección manual.
        // Dado que RiverState es un record/DTO, llenamos arrays dummy para T, pH, etc.
        RiverState initialState = new RiverState(
                initialDepth, initialVelocity, zeroArray, zeroArray, zeroArray
        );

        // 2. Input del Batch (1D Array - Smart Fetch)
        float[] flowInput = new float[BATCH_SIZE];
        Arrays.fill(flowInput, 150.0f); // Caudal entrante constante

        // 3. Temperatura/pH Dummy (No afectan a la hidráulica en este test)
        float[][][] phTmp = new float[BATCH_SIZE][2][cellCount];

        ManningSimulationResult resultCpu;
        ManningSimulationResult resultGpu;

        // --- EJECUCIÓN A: MODO CPU (Legacy Logic) ---
        log.info(">> Ejecutando en CPU...");

        long t1 = System.nanoTime();
        // Usamos try-with-resources para asegurar limpieza
        try (ManningBatchProcessor cpuProcessor = new ManningBatchProcessor(realGeometry, cpuConfig)) {
            // El procesador expandirá la matriz internamente
            resultCpu = cpuProcessor.processBatch(
                    BATCH_SIZE, initialState, flowInput, phTmp,
                    false // Force CPU
            );
        }
        long cpuTime = System.nanoTime() - t1;

        // --- EJECUCIÓN B: MODO GPU (Smart Fetch Logic) ---
        log.info(">> Ejecutando en GPU...");

        long t2 = System.nanoTime();
        try (ManningBatchProcessor gpuProcessor = new ManningBatchProcessor(realGeometry, gpuConfig)) {
            // El procesador enviará el array comprimido a la VRAM
            resultGpu = gpuProcessor.processBatch(
                    BATCH_SIZE, initialState, flowInput, phTmp,
                    true // Force GPU
            );
        }
        long gpuTime = System.nanoTime() - t2;

        log.info("Tiempos (Incluyendo Init) -> CPU: {}ms | GPU: {}ms", cpuTime/1e6, gpuTime/1e6);

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

            // Usamos una tolerancia relativa si el valor es grande, o absoluta si es pequeño
            if (Math.abs(hCpu - hGpu) > EPSILON) {
                fail(String.format("Divergencia H en T=%d C=%d. CPU=%.5f GPU=%.5f Delta=%.5f",
                        step, i, hCpu, hGpu, Math.abs(hCpu - hGpu)));
            }

            // Velocidad
            double vCpu = cpu.getVelocityAt(i);
            double vGpu = gpu.getVelocityAt(i);
            if (Math.abs(vCpu - vGpu) > EPSILON) {
                fail(String.format("Divergencia V en T=%d C=%d. CPU=%.5f GPU=%.5f Delta=%.5f",
                        step, i, vCpu, vGpu, Math.abs(vCpu - vGpu)));
            }
        }
    }
}