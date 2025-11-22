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
import projectstalker.physics.model.RiverPhModel;
import projectstalker.physics.model.RiverTemperatureModel;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.mock;

/**
 * Test de Integración End-to-End para el solver de GPU.
 * Este test carga la librería nativa real (.so) y ejecuta una simulación
 * a través del stack JNI -> C++ -> CUDA.
 * Requiere la tarea 'gpuTest' de Gradle para ejecutarse, ya que necesita
 * que 'projectstalker.native.enabled' y 'java.library.path' estén configurados.
 */
@Tag("GPU") // Etiqueta para que solo la tarea 'gpuTest' lo ejecute
@Slf4j
class ManningGpuIntegrationTest {

    private ManningBatchProcessor batchProcessor;
    private RiverGeometry realGeometry;
    private RiverTemperatureModel realTempModel;
    private RiverPhModel realPhModel;
    private SimulationConfig simConfig;

    // Dimensiones (igual que en el test de CPU)
    private int cellCount;
    private final int BATCH_SIZE = 3;
    private final double DELTA_TIME = 10.0;

    @BeforeEach
    void setUp() {
        log.info("Configurando entorno para Test de Integración GPU...");
        // --- 1. Inicializar INSTANCIA REAL de Geometría y Configuración ---
        RiverConfig riverConfig = RiverConfig.getTestingRiver();
        RiverGeometryFactory factory = new RiverGeometryFactory();
        this.realGeometry = factory.createRealisticRiver(riverConfig);

        // --- 2. Inicializar INSTANCIAS REALES de Modelos Fisicoquímicos ---
        this.realTempModel = new RiverTemperatureModel(riverConfig, this.realGeometry);
        this.realPhModel = new RiverPhModel(this.realGeometry);

        // --- 3. Inicializar Mocks de Configuración ---
        simConfig = mock(SimulationConfig.class);
        this.simConfig = SimulationConfig.builder()
                .riverConfig(riverConfig)
                .seed(12345L)
                .totalTime(3600) // 1 hora
                .deltaTime((float) DELTA_TIME)
                .cpuProcessorCount(Runtime.getRuntime().availableProcessors())
                .cpuTimeBatchSize(BATCH_SIZE) // Sincronizado con la constante del test
                .useGpuAccelerationOnManning(true) // Intención explícita
                .useGpuAccelerationOnTransport(false) // Por ahora false
                .build();
        this.cellCount = this.realGeometry.getCellCount();
        log.info("Entorno configurado. Geometría con {} celdas.", this.realGeometry.getCellCount());
    }

    @Test
    @DisplayName("El batch en modo GPU debe cargar la librería nativa y ejecutar sin crashear")
    void processBatch_gpuMode_shouldRunOnNativeLibraryWithoutCrashing() {
        log.info("Iniciando test de integración de GPU. BATCH_SIZE={}", BATCH_SIZE);

        // --- ARRANGE: Preparación de Datos ---
        double currentTime = 300.0 * 3600.0;

        // 1. Estado Inicial
        double initialUniformDepth = 0.5;
        double[] initialData = new double[cellCount];
        Arrays.fill(initialData, initialUniformDepth);
        RiverState initialRiverState = new RiverState(
                initialData, initialData, initialData, initialData
        );

        // 2. Perfiles de Caudal
        double[] newDischarges = new double[BATCH_SIZE];
        Arrays.fill(newDischarges, 200.0);
        double[] initialDischarges = new double[cellCount];
        Arrays.fill(initialDischarges, 50);

        // 3. Instanciar el SUT (Subject Under Test) REAL
        // Esta línea es la que disparará la carga de la librería nativa
        // (porque 'gpuTest' define 'projectstalker.native.enabled=true')
        log.info("Instanciando ManningBatchProcessor (esto debería cargar la librería nativa)...");
        batchProcessor = assertDoesNotThrow(
                () -> new ManningBatchProcessor(this.realGeometry, simConfig),
                "Falló la instanciación de ManningBatchProcessor. ¿Error al cargar la librería nativa?"
        );
        log.info("ManningBatchProcessor instanciado. Librería nativa cargada.");

        // (Re-usamos el método del SUT para crear los perfiles)
        double[][] allDischargeProfiles = batchProcessor.createDischargeProfiles(BATCH_SIZE, newDischarges, initialDischarges);

        // 4. Resultados Fisicoquímicos (Pre-cálculo)
        double[][][] phTmp = new double[BATCH_SIZE][2][cellCount];
        double timeStep = 0.0;
        for (int i = 0; i < BATCH_SIZE; i++) {
            double t = currentTime + timeStep;
            phTmp[i][0] = realTempModel.calculate(t);
            phTmp[i][1] = realPhModel.getPhProfile();
            timeStep += DELTA_TIME;
        }

        // --- ACT & ASSERT (Prueba de Supervivencia) ---
        log.warn(">>> EJECUTANDO STACK NATIVO (GPU) <<<");

        // Esta es la prueba principal:
        // Verificamos que la llamada a processBatch (con GPU=true)
        // se completa sin lanzar ninguna excepción de JNI, C++, o CUDA.
        ManningSimulationResult result = assertDoesNotThrow(() ->
                        batchProcessor.processBatch(
                                BATCH_SIZE, initialRiverState,
                                allDischargeProfiles, phTmp, true // <-- ¡GPU HABILITADA!
                        ),
                "La ejecución nativa de la GPU falló (lanzó una excepción)"
        );

        log.warn(">>> STACK NATIVO EJECUTADO EXITOSAMENTE <<<");

        // --- ASSERT (Verificaciones de Cordura) ---
        // Si llegamos aquí, el stack nativo funcionó.
        // Ahora solo verificamos que los datos de vuelta no son basura.
        assertNotNull(result, "El resultado de la GPU no debe ser nulo.");
        assertEquals(BATCH_SIZE, result.getStates().size(), "El resultado debe tener el tamaño de batch correcto.");

        RiverState finalState = result.getStates().get(BATCH_SIZE - 1);
        assertNotNull(finalState, "El estado final es nulo.");

        // Verificamos que no hay NaNs
        assertFalse(Double.isNaN(finalState.getWaterDepthAt(10)), "El resultado de profundidad de la GPU es NaN.");
        assertFalse(Double.isNaN(finalState.getVelocityAt(10)), "El resultado de velocidad de la GPU es NaN.");

        // Verificamos que la GPU realmente hizo un cálculo (no devolvió ceros)
        assertTrue(finalState.getWaterDepthAt(10) > 0.0, "La GPU no calculó una profundidad positiva.");
        assertTrue(finalState.getVelocityAt(10) > 0.0, "La GPU no calculó una velocidad positiva.");

        log.info("Verificaciones de cordura (Sanity Checks) superadas. Test de GPU exitoso.");
    }
}