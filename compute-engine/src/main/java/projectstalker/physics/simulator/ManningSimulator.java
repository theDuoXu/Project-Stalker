package projectstalker.physics.simulator;

import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import projectstalker.config.RiverConfig;
import projectstalker.config.SimulationConfig;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;
import projectstalker.domain.simulation.IManningResult;
import projectstalker.factory.RiverGeometryFactory;
import projectstalker.factory.RiverFactory;
import projectstalker.physics.model.*;

/**
 * Orquesta la simulación hidrológica del río.
 * Facade de alto nivel que prepara los datos iniciales y delega la ejecución
 * intensiva al {@link ManningBatchProcessor}.
 * <p>
 * REFACTORIZADO: Ahora actúa como un lanzador de "Ejecución Completa" en lugar de un stepper,
 * aprovechando la capacidad del BatchProcessor para gestionar la memoria y la GPU internamente.
 */
@Slf4j
public class ManningSimulator implements AutoCloseable {

    @Getter
    private final RiverGeometry geometry;
    private final SimulationConfig simulationConfig;

    private final RandomFlowProfileGenerator flowGenerator;

    private final ManningBatchProcessor batchProcessor;

    @Getter @Setter
    private RiverState initialState;

    public ManningSimulator(RiverConfig riverConfig, SimulationConfig simulationConfig) {
        this.simulationConfig = simulationConfig;

        // 1. Construcción del Dominio Físico
        this.geometry = RiverGeometryFactory.createRealisticRiver(riverConfig);

        // 2. Modelos Auxiliares (Condiciones de Frontera)
        this.flowGenerator = new RandomFlowProfileGenerator((int) riverConfig.seed(), simulationConfig.getFlowConfig());
        // 3. Estado Inicial (t=0)
        // Usamos la factoría para crear un río estable o vacío según se requiera
        this.initialState = RiverFactory.createSteadyState(this.geometry, simulationConfig.getFlowConfig().getBaseDischarge());

        // 4. Inicialización del Motor Físico (Orquestador)
        this.batchProcessor = new ManningBatchProcessor(this.geometry, simulationConfig);

        log.info("ManningSimulator listo. (GPU: {}, Strategy: {})",
                simulationConfig.isUseGpuAccelerationOnManning(),
                simulationConfig.getGpuStrategy());
    }

    /**
     * Ejecuta la simulación completa basada en la configuración temporal.
     * <p>
     * Genera el hidrograma de entrada completo y lo envía al procesador.
     * El procesador decidirá internamente cómo dividir el trabajo (Chunks/Strides)
     * y retornará un resultado optimizado en memoria.
     * * @return El resultado completo de la simulación (IManningResult).
     */
    public IManningResult runFullSimulation() {
        log.info("Iniciando simulación completa...");

        // 1. Generación del Perfil de Entrada (Hidrograma)
        // Calculamos Q_inflow para cada delta_t de toda la simulación.
        float[] fullInflowProfile = generateFullInflowProfile();

        log.info("Hidrograma generado: {} pasos de tiempo.", fullInflowProfile.length);

        // 2. Delegación al Núcleo Físico
        // El procesador gestiona la complejidad de GPU, memoria y paginación.
        IManningResult result = batchProcessor.process(fullInflowProfile, this.initialState);

        log.info("Simulación finalizada. Tiempo de cómputo: {}ms", result.getSimulationTime());
        return result;
    }

    /**
     * Genera el array de caudales de entrada para toda la duración configurada.
     */
    private float[] generateFullInflowProfile() {
        long totalSteps = simulationConfig.getTotalTimeSteps();

        // Validación de seguridad de memoria para el perfil de entrada
        if (totalSteps > Integer.MAX_VALUE - 100) {
            throw new IllegalArgumentException("La simulación es demasiado larga para indexar en un array Java (Max ~2B pasos).");
        }

        int steps = (int) totalSteps;
        float[] inflows = new float[steps];
        float dt = simulationConfig.getDeltaTime();
        double time = 0.0;

        for (int i = 0; i < steps; i++) {
            inflows[i] = flowGenerator.getDischargeAt(time);
            time += dt;
        }

        return inflows;
    }

    @Override
    public void close() {
        if (batchProcessor != null) {
            batchProcessor.close();
        }
        log.info("ManningSimulator cerrado y recursos liberados.");
    }
}