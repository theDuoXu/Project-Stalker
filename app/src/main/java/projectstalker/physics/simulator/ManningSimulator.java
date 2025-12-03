package projectstalker.physics.simulator;

import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import projectstalker.config.RiverConfig;
import projectstalker.config.SimulationConfig;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;
import projectstalker.domain.simulation.ManningSimulationResult;
import projectstalker.factory.RiverGeometryFactory;
import projectstalker.physics.model.FlowProfileModel;
import projectstalker.physics.model.RiverPhModel;
import projectstalker.physics.model.RiverTemperatureModel;

/**
 * Orquesta la simulación hidrológica del río.
 * <p>
 * REFACTORIZADO: Ahora actúa como una fachada de alto nivel sobre {@link ManningBatchProcessor}.
 * Gestiona el tiempo global, la generación de condiciones de frontera (caudales, T, pH)
 * y el estado actual del río.
 * <p>
 * Implementa AutoCloseable para cerrar limpiamente la sesión de GPU subyacente.
 */
@Slf4j
public class ManningSimulator implements AutoCloseable {

    // --- Miembros de la Simulación ---
    private final RiverConfig config;
    @Getter
    private final RiverGeometry geometry;

    // Modelos de condiciones de frontera
    private final FlowProfileModel flowGenerator;
    private final RiverTemperatureModel temperatureModel;
    private final RiverPhModel phModel;

    // EL MOTOR ÚNICO: Procesa tanto pasos simples como batches
    private final ManningBatchProcessor batchProcessor;

    // --- Estado de la Simulación ---
    @Getter
    @Setter
    private RiverState currentState;
    @Getter
    @Setter
    private double currentTimeInSeconds;
    @Getter
    @Setter
    private boolean isGpuAccelerated;

    /**
     * Constructor del simulador.
     *
     * @param config           La configuración global para el río.
     * @param simulationConfig La configuración para la ejecución de la simulación.
     */
    public ManningSimulator(RiverConfig config, SimulationConfig simulationConfig) {
        this.config = config;
        this.geometry = new RiverGeometryFactory().createRealisticRiver(config);

        // Inicialización de modelos físicos
        this.flowGenerator = new FlowProfileModel((int) config.seed(), simulationConfig.getFlowConfig());
        this.temperatureModel = new RiverTemperatureModel(config, this.geometry);
        this.phModel = new RiverPhModel(this.geometry);

        // Estado inicial (t=0)
        int cellCount = geometry.getCellCount();
        this.currentState = new RiverState(new float[cellCount], new float[cellCount], new float[cellCount], new float[cellCount], new float[cellCount]);
        this.currentTimeInSeconds = 0.0;
        this.isGpuAccelerated = simulationConfig.isUseGpuAccelerationOnManning();

        // Inicialización del BatchProcessor (Crea Session GPU + ThreadPool CPU)
        this.batchProcessor = new ManningBatchProcessor(this.geometry, simulationConfig);

        log.info("ManningSimulator inicializado correctamente (Unified Batch Processor).");
    }

    /**
     * Avanza la simulación un único paso de tiempo.
     * <p>
     * Se implementa delegando al BatchProcessor con un tamaño de batch de 1.
     * Esto simplifica el mantenimiento al tener una única ruta de código para la física.
     *
     * @param deltaTimeInSeconds El incremento de tiempo en segundos.
     */
    public void advanceTimeStep(double deltaTimeInSeconds) {
        advanceBatchTimeStep(deltaTimeInSeconds, 1);
    }

    /**
     * Avanza la simulación para un batch de pasos de tiempo.
     *
     * @param deltaTimeInSeconds El incremento de tiempo para cada paso del batch.
     * @param batchSize          El número de pasos a simular.
     * @return El resultado de la simulación para el batch completo.
     */
    public ManningSimulationResult advanceBatchTimeStep(double deltaTimeInSeconds, int batchSize) {
        if (batchSize <= 0) {
            throw new IllegalArgumentException("El tamaño del batch debe ser mayor que 0.");
        }

        int cellCount = geometry.getCellCount();
        RiverState initialRiverState = this.currentState;

        // 1. PRE-CÓMPUTO DE CONDICIONES DE FRONTERA (Inputs)
        // Generamos arrays compactos [BatchSize] en lugar de matrices expandidas.
        float[] newDischarges = new float[batchSize];
        float[][][] phTmp = new float[batchSize][2][cellCount]; // [t][0]=T, [t][1]=pH

        // Simulamos el avance del tiempo solo para generar los inputs
        double tempTime = currentTimeInSeconds;

        for (int i = 0; i < batchSize; i++) {
            // Caudal de entrada en t (Input para Smart Fetch)
            newDischarges[i] = flowGenerator.getDischargeAt(tempTime);

            // Temperatura y pH
            float[] temp = temperatureModel.calculate(tempTime);
            float[] ph = phModel.getPhProfile();
            phTmp[i][0] = temp;
            phTmp[i][1] = ph;

            tempTime += deltaTimeInSeconds;
        }

        // 2. EJECUCIÓN DELEGADA (Híbrida CPU/GPU)
        // Pasamos los inputs compactos. El Processor decide si expandir en VRAM (GPU) o RAM (CPU).
        ManningSimulationResult result = batchProcessor.processBatch(
                batchSize,
                initialRiverState,
                newDischarges,
                phTmp,
                isGpuAccelerated
        );

        // 3. ACTUALIZACIÓN DE ESTADO
        // Confirmamos el avance del tiempo y guardamos el último estado calculado.
        this.currentTimeInSeconds = tempTime;

        if (!result.getStates().isEmpty()) {
            this.currentState = result.getStates().get(batchSize - 1);
        }

        return result;
    }

    /**
     * Devuelve el caudal de entrada para el tiempo actual de la simulación.
     */
    public float getCurrentInputDischarge() {
        return flowGenerator.getDischargeAt(currentTimeInSeconds);
    }

    /**
     * Cierre limpio de recursos.
     * Cierra la sesión GPU asociada al procesador.
     */
    @Override
    public void close() {
        if (batchProcessor != null) {
            batchProcessor.close();
            log.info("ManningSimulator cerrado y recursos liberados.");
        }
    }
}