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
     * El número de pasos de tiempo (Δt) que se agrupan y se pasan a la GPU para cómputo en un solo lote.
     * Este valor gestiona el compromiso entre el overhead de la transferencia de datos CPU-GPU
     * y la latencia (responsiveness) del sistema.
     */
    int cpuTimeBatchSize;

    /**
     * Número de núcleos CPU a utilizar en simulaciones CPU
     */
    int cpuProcessorCount;

    public long getTotalTimeSteps() {
        return Math.round(totalTime / deltaTime);
    }

    /**
     * Define los parámetros para el generador de caudal de entrada.
     */
    @Value
    @Builder
    public static class FlowConfig {
        /**
         * Caudal promedio o base sobre el cual se aplican las variaciones.
         */
        double baseDischarge;
        /**
         * Magnitud máxima de la variación sobre el caudal base.
         */
        double noiseAmplitude;
        /**
         * Frecuencia de la variación (valores bajos para cambios lentos).
         */
        float noiseFrequency;
    }
}