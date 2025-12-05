package projectstalker.physics.simulator;

import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import projectstalker.config.RiverConfig;
import projectstalker.config.SimulationConfig;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;
import projectstalker.domain.simulation.ISimulationResult;
import projectstalker.factory.RiverGeometryFactory;
import projectstalker.physics.model.FlowProfileModel;
import projectstalker.physics.model.RiverPhModel;
import projectstalker.physics.model.RiverTemperatureModel;
import projectstalker.config.SimulationConfig.GpuStrategy;

/**
 * Orquesta la simulación hidrológica del río.
 * Facade de alto nivel sobre {@link ManningBatchProcessor}.
 */
@Slf4j
public class ManningSimulator implements AutoCloseable {

    private final RiverConfig config;
    @Getter
    private final RiverGeometry geometry;

    private final FlowProfileModel flowGenerator;
    private final RiverTemperatureModel temperatureModel;
    private final RiverPhModel phModel;

    private final ManningBatchProcessor batchProcessor;

    @Getter @Setter
    private RiverState currentState;
    @Getter @Setter
    private double currentTimeInSeconds;
    @Getter @Setter
    private boolean isGpuAccelerated;

    // Configuración de Estrategia por defecto
    private final GpuStrategy defaultGpuStrategy;

    public ManningSimulator(RiverConfig config, SimulationConfig simulationConfig) {
        this.config = config;
        this.geometry = new RiverGeometryFactory().createRealisticRiver(config);

        this.flowGenerator = new FlowProfileModel((int) config.seed(), simulationConfig.getFlowConfig());
        this.temperatureModel = new RiverTemperatureModel(config, this.geometry);
        this.phModel = new RiverPhModel(this.geometry);

        int cellCount = geometry.getCellCount();
        this.currentState = new RiverState(new float[cellCount], new float[cellCount], new float[cellCount], new float[cellCount], new float[cellCount]);
        this.currentTimeInSeconds = 0.0;

        this.isGpuAccelerated = simulationConfig.isUseGpuAccelerationOnManning();

        // MAPEO DE CONFIGURACIÓN -> ESTRATEGIA INTERNA
        // Convertimos el Enum de Configuración (DTO) al Enum del Procesador (Lógica)
        this.defaultGpuStrategy = mapConfigStrategy(simulationConfig.getGpuStrategy());

        this.batchProcessor = new ManningBatchProcessor(this.geometry, simulationConfig);

        log.info("ManningSimulator inicializado. GPU={}, Strategy={}", isGpuAccelerated, defaultGpuStrategy);
    }

    /**
     * Mapea el Enum de configuración al Enum interno del procesador.
     * Esto desacopla la capa de configuración de la capa física.
     */
    private GpuStrategy mapConfigStrategy(SimulationConfig.GpuStrategy configStrategy) {
        if (configStrategy == null) return GpuStrategy.SMART_SAFE;

        switch (configStrategy) {
            case FULL_EVOLUTION: return GpuStrategy.FULL_EVOLUTION;
            case SMART_TRUSTED:  return GpuStrategy.SMART_TRUSTED;
            case SMART_SAFE:
            default:             return GpuStrategy.SMART_SAFE;
        }
    }

    public void advanceTimeStep(double deltaTimeInSeconds) {
        advanceBatchTimeStep(deltaTimeInSeconds, 1);
    }

    public ISimulationResult advanceBatchTimeStep(double deltaTimeInSeconds, int batchSize) {
        // Usa la estrategia definida en SimulationConfig
        return advanceBatchTimeStep(deltaTimeInSeconds, batchSize, this.defaultGpuStrategy);
    }

    public ISimulationResult advanceBatchTimeStep(double deltaTimeInSeconds, int batchSize, GpuStrategy strategy) {
        if (batchSize <= 0) throw new IllegalArgumentException("BatchSize debe ser > 0");

        int cellCount = geometry.getCellCount();

        float[] newDischarges = new float[batchSize];
        float[][][] phTmp = new float[batchSize][2][cellCount];

        double tempTime = currentTimeInSeconds;

        for (int i = 0; i < batchSize; i++) {
            newDischarges[i] = flowGenerator.getDischargeAt(tempTime);

            phTmp[i][0] = temperatureModel.calculate(tempTime);
            phTmp[i][1] = phModel.getPhProfile();

            tempTime += deltaTimeInSeconds;
        }

        ISimulationResult result = batchProcessor.processBatch(
                batchSize,
                this.currentState,
                newDischarges,
                phTmp,
                this.isGpuAccelerated,
                strategy // Pasamos la estrategia seleccionada (Full/Smart)
        );

        this.currentTimeInSeconds = tempTime;

        if (result.getFinalState().isPresent()) {
            this.currentState = result.getFinalState().get();
        } else {
            this.currentState = result.getStateAt(batchSize - 1);
        }

        return result;
    }

    public float getCurrentInputDischarge() {
        return flowGenerator.getDischargeAt(currentTimeInSeconds);
    }

    @Override
    public void close() {
        if (batchProcessor != null) {
            batchProcessor.close();
            log.info("ManningSimulator cerrado.");
        }
    }
}