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
import projectstalker.domain.simulation.ISimulationResult;
import projectstalker.factory.RiverGeometryFactory;
import projectstalker.physics.impl.ManningProfileCalculatorTask; // Importamos la Tarea de CPU

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

@Tag("GPU")
@Slf4j
class ManningGpuAccuracyTest {

    private RiverGeometry realGeometry;
    private SimulationConfig cpuConfig;
    private SimulationConfig gpuConfig;

    private int cellCount;
    private final int BATCH_SIZE = 5;
    // Tolerancia relajada para Float (GPU/Fast Math) vs Double (CPU/Precise)
    // Manning tiene potencias fraccionarias, así que el error se acumula.
    private final float EPSILON = 5e-3f;

    // Estado inicial en Equilibrio Hidráulico
    private RiverState stableInitialState;
    private final float BASE_DISCHARGE = 50.0f; // Caudal base estable

    @BeforeEach
    void setUp() throws Exception {
        // 1. Geometría Real
        RiverConfig riverConfig = RiverConfig.getTestingRiver();
        RiverGeometryFactory factory = new RiverGeometryFactory();
        this.realGeometry = factory.createRealisticRiver(riverConfig);
        this.cellCount = this.realGeometry.getCellCount();

        log.info("Test configurado. Geometría real tiene {} celdas.", this.cellCount);

        // 2. GENERAR ESTADO INICIAL ESTABLE (Warm-Up)
        // Usamos la implementación de CPU existente para calcular el perfil H para un Q constante.

        float[] qProfile = new float[cellCount];
        Arrays.fill(qProfile, BASE_DISCHARGE);

        // Semilla para Newton-Raphson (valor razonable)
        float[] seedDepth = new float[cellCount];
        Arrays.fill(seedDepth, 1.0f);

        // Instanciamos y ejecutamos la tarea de cálculo síncronamente
        ManningProfileCalculatorTask calculator = new ManningProfileCalculatorTask(
                qProfile, seedDepth, realGeometry
        );

        // call() ejecuta la lógica y guarda el resultado internamente
        calculator.call();

        // Construimos el estado estable con los resultados calculados
        // Ahora H y V son consistentes con Q y la Geometría.
        this.stableInitialState = new RiverState(
                calculator.getCalculatedWaterDepth(), // H equilibrada
                calculator.getCalculatedVelocity(),   // V equilibrada
                new float[cellCount], // T (dummy)
                new float[cellCount], // pH (dummy)
                new float[cellCount]  // C (dummy)
        );

        log.info("Estado estable generado. Q={}, H_entrada={}",
                BASE_DISCHARGE, calculator.getCalculatedWaterDepth()[0]);

        // 3. Configuración del Simulador
        SimulationConfig baseConfig = SimulationConfig.builder()
                .riverConfig(riverConfig)
                .seed(12345L)
                .totalTime(3600)
                .deltaTime(10.0f)
                .cpuProcessorCount(4)
                .useGpuAccelerationOnTransport(false)
                .build();

        this.cpuConfig = baseConfig.withUseGpuAccelerationOnManning(false);
        this.gpuConfig = baseConfig.withUseGpuAccelerationOnManning(true);
    }

    @Test
    @DisplayName("Paridad Numérica: GPU (Flyweight) debe coincidir con CPU (Dense) partiendo de equilibrio")
    void compareCpuVsGpu_shouldProduceIdenticalResults() {
        log.info("=== INICIANDO TEST DE PRECISIÓN NUMÉRICA ===");

        // 1. Usamos el estado estable generado en setUp
        RiverState initialState = this.stableInitialState;

        // 2. Input del Batch (Perturbación / Onda de Avenida)
        // Introducimos un caudal mucho mayor para generar una onda dinámica
        float[] flowInput = new float[BATCH_SIZE];
        Arrays.fill(flowInput, 150.0f); // 150 vs 50 base

        // 3. Dummy Aux
        float[][][] phTmp = new float[BATCH_SIZE][2][cellCount];

        ISimulationResult resultCpu;
        ISimulationResult resultGpu;

        // --- EJECUCIÓN CPU (Referencia) ---
        log.info(">> Ejecutando en CPU...");
        try (ManningBatchProcessor cpuProcessor = new ManningBatchProcessor(realGeometry, cpuConfig)) {
            resultCpu = cpuProcessor.processBatch(BATCH_SIZE, initialState, flowInput, phTmp, false);
        }

        // --- EJECUCIÓN GPU (SUT) ---
        log.info(">> Ejecutando en GPU...");
        try (ManningBatchProcessor gpuProcessor = new ManningBatchProcessor(realGeometry, gpuConfig)) {
            resultGpu = gpuProcessor.processBatch(BATCH_SIZE, initialState, flowInput, phTmp, true);
        }

        // --- VALIDACIÓN ---
        for (int t = 0; t < BATCH_SIZE; t++) {
            // Obtenemos los estados usando la interfaz común.
            // CPU: Lee de memoria. GPU: Reconstruye usando Flyweight.
            RiverState sCpu = resultCpu.getStateAt(t);
            RiverState sGpu = resultGpu.getStateAt(t);

            compareStates(t, sCpu, sGpu);
        }
        log.info("=== TEST DE PARIDAD SUPERADO CON ÉXITO ===");
    }

    private void compareStates(int step, RiverState cpu, RiverState gpu) {
        // Limitamos la comprobación a la zona de interés + margen.
        // El Flyweight asume que más allá de 'step', el estado es igual al inicial desplazado.
        // Como partimos de equilibrio (InitialState estable), InitialState desplazado == InitialState estático.
        // Por tanto, la CPU (que recalcula todo) debería dar el mismo valor.

        int checkLimit = Math.min(cellCount, BATCH_SIZE + 50);

        for (int i = 0; i < checkLimit; i++) {
            double hCpu = cpu.getWaterDepthAt(i);
            double hGpu = gpu.getWaterDepthAt(i);

            if (Math.abs(hCpu - hGpu) > EPSILON) {
                fail(String.format("Divergencia H en T=%d C=%d. CPU=%.5f GPU=%.5f Delta=%.5f",
                        step, i, hCpu, hGpu, Math.abs(hCpu - hGpu)));
            }

            double vCpu = cpu.getVelocityAt(i);
            double vGpu = gpu.getVelocityAt(i);

            if (Math.abs(vCpu - vGpu) > EPSILON) {
                fail(String.format("Divergencia V en T=%d C=%d. CPU=%.5f GPU=%.5f Delta=%.5f",
                        step, i, vCpu, vGpu, Math.abs(vCpu - vGpu)));
            }
        }
    }
}