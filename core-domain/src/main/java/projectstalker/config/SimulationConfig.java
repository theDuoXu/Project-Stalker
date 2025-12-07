package projectstalker.config;

import lombok.Builder;
import lombok.Value;
import lombok.With;

/**
 * Contenedor principal para todas las configuraciones de una simulación.
 * Agrupa la configuración física del río con los parámetros de la simulación.
 */
@Value
@Builder
@With
public class SimulationConfig {

    /**
     * Configuración de las propiedades físicas y geomorfológicas del río.
     */
    RiverConfig riverConfig;

    /**
     * Semilla principal para todos los generadores de ruido de la simulación.
     */
    long seed;

    /**
     * Configuración del generador de perfil de caudal de entrada.
     */
    FlowConfig flowConfig;

    /**
     * Configuración temporal del simulador
     */
    float totalTime;
    /**
     * Paso de tiempo
     */
    float deltaTime;

    /**
     * Utiliza aceleración CUDA para estimar la velocidad y la profundidad del agua
     */
    boolean useGpuAccelerationOnManning;

    /**
     * Utiliza aceleración CUDA para estimar las soluciones EDP advección difusión reacción
     */
    boolean useGpuAccelerationOnTransport;

    /**
     * Define la estrategia de optimización y física para el solver Manning en GPU.
     * Por defecto: SMART_SAFE (Optimizado con verificación de seguridad).
     */
    @Builder.Default
    GpuStrategy gpuStrategy = GpuStrategy.SMART_SAFE;

    /**
     * El número de pasos de tiempo (Δt) que se agrupan y se pasan a la GPU.
     */
    int cpuTimeBatchSize;

    /**
     * El factor de submuestreo a aplicar
     */
    @Builder.Default
    int gpuFullEvolutionStride = 1;
    /**
     * Número de núcleos CPU a utilizar en simulaciones CPU
     */
    int cpuProcessorCount;

    public long getTotalTimeSteps() {
        return Math.round(totalTime / deltaTime);
    }

    /**
     * Estrategias disponibles para la ejecución GPU.
     */
    public enum GpuStrategy {
        /**
         * Optimización agresiva. Asume Steady State. Descarga triangular.
         * Más rápido, pero requiere que el río base esté estabilizado.
         */
        SMART_TRUSTED,

        /**
         * Optimización segura (Default). Verifica caudal estable.
         * Si el río es inestable, hace fallback automático a FULL_EVOLUTION.
         */
        SMART_SAFE,

        /**
         * Simulación completa robusta.
         * Calcula y descarga todo el dominio. Sin asunciones. Más lento en transferencia (PCIe).
         */
        FULL_EVOLUTION
    }

    /**
     * Define los parámetros para el generador de caudal de entrada.
     */
    @Value
    @Builder
    public static class FlowConfig {
        float baseDischarge;
        float noiseAmplitude;
        float noiseFrequency;
    }
}