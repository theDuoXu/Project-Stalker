package projectstalker.config;

import lombok.Builder;
import lombok.Value;

/**
 * Contenedor principal para todas las configuraciones de una simulación.
 * Agrupa la configuración física del río con los parámetros de la simulación.
 */
@Value
@Builder
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