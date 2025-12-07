package projectstalker.physics.simulator;

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
import projectstalker.domain.simulation.IManningResult;
import projectstalker.factory.RiverFactory;
import projectstalker.factory.RiverGeometryFactory;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test de Integración End-to-End para el solver de GPU.
 * <p>
 * Verifica la carga de librería nativa (JNI/CUDA) y la ejecución del ciclo completo
 * utilizando la nueva arquitectura de {@link ManningBatchProcessor}.
 */
@Tag("GPU")
@Slf4j
class ManningGpuIntegrationTest {

    private RiverGeometry realGeometry;
    private SimulationConfig baseConfig;

    private int cellCount;
    private final int BATCH_SIZE = 5;
    private final double DELTA_TIME = 10.0;

    @BeforeEach
    void setUp() {
        log.info("Configurando entorno para Test de Integración GPU...");

        RiverConfig riverConfig = RiverConfig.getTestingRiver();
        RiverGeometryFactory factory = new RiverGeometryFactory();
        this.realGeometry = factory.createRealisticRiver(riverConfig);
        this.cellCount = this.realGeometry.getCellCount();

        // Configuración Base (Común)
        this.baseConfig = SimulationConfig.builder()
                .riverConfig(riverConfig)
                .seed(12345L)
                .totalTime(3600)
                .deltaTime((float) DELTA_TIME)
                .cpuProcessorCount(2)
                .cpuTimeBatchSize(BATCH_SIZE) // Alineado con el input de prueba
                .gpuFullEvolutionStride(1)    // Stride 1 para validación simple
                .useGpuAccelerationOnManning(true) // Activar GPU
                .useGpuAccelerationOnTransport(false)
                .build();

        log.info("Entorno configurado. Geometría con {} celdas.", this.realGeometry.getCellCount());
    }

    @Test
    @DisplayName("Integration SMART: El modo optimizado debe ejecutar sin crashear")
    void integrationTest_SmartMode() {
        runIntegrationTest(GpuStrategy.SMART_TRUSTED);
    }

    @Test
    @DisplayName("Integration FULL: El modo robusto debe ejecutar sin crashear")
    void integrationTest_FullMode() {
        runIntegrationTest(GpuStrategy.FULL_EVOLUTION);
    }

    private void runIntegrationTest(GpuStrategy strategy) {
        log.info(">>> INICIANDO INTEGRATION TEST: {} <<<", strategy);

        // 1. Estado Inicial Estable (Vital para Smart Mode)
        // Usamos la Factory analítica para garantizar Q_in = Q_out inicial
        RiverState initialRiverState = RiverFactory.createSteadyState(this.realGeometry, 50.0f);

        // 2. Inputs (Hidrograma simple)
        float[] newDischarges = new float[BATCH_SIZE];
        Arrays.fill(newDischarges, 200.0f); // Onda de avenida

        // 3. Configurar Estrategia Específica
        SimulationConfig specificConfig = baseConfig.withGpuStrategy(strategy);

        // 4. Instanciar SUT y Ejecutar (Try-with-resources para asegurar cierre JNI)
        // assertDoesNotThrow envuelve todo el ciclo de vida
        IManningResult result = assertDoesNotThrow(() -> {
            try (ManningBatchProcessor processor = new ManningBatchProcessor(this.realGeometry, specificConfig)) {
                return processor.process(newDischarges, initialRiverState);
            }
        }, "Fallo crítico en ejecución nativa JNI/CUDA");

        // --- ASSERT ---
        assertNotNull(result, "El resultado no debe ser nulo");

        // Verificación de integridad básica
        assertEquals(BATCH_SIZE, result.getTimestepCount(), "El número de pasos simulados debe coincidir con el input");

        // Obtenemos el último estado para verificar sanidad numérica
        RiverState finalState = result.getStateAt(BATCH_SIZE - 1);

        float hHead = finalState.getWaterDepthAt(0);
        float hTail = finalState.getWaterDepthAt(cellCount - 1);

        // Chequeo de sanidad (No NaN, No Infinito, No Negativo)
        assertFalse(Float.isNaN(hHead), "El resultado contiene NaNs (Head)");
        assertTrue(hHead > 0, "La profundidad debe ser positiva (Head)");

        assertFalse(Float.isNaN(hTail), "El resultado contiene NaNs (Tail)");
        assertTrue(hTail > 0, "La profundidad debe ser positiva (Tail)");

        log.info(">>> TEST {} SUPERADO. H_final(0) = {} m <<<", strategy, String.format("%.3f", hHead));
    }
}