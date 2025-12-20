package projectstalker.config;

import com.fasterxml.jackson.annotation.JsonView;
import lombok.Builder;
import lombok.With;
import projectstalker.domain.sensors.SensorViews;

/**
 * Un objeto de valor inmutable para contener todos los parámetros de configuración
 * necesarios para la generación procedural de un río.
 * <p>
 * Esta clase agrupa todas las variables de alto nivel que definen la morfología,
 * hidrología y las propiedades físico-químicas básicas del río a simular.
 *
 * @param seed                     Semilla para la generación de ruido, para resultados reproducibles.
 * @param noiseFrequency           Controla el nivel de detalle y la escala de las características del ruido.
 * @param detailNoiseFrequency     Frecuencia del ruido para variaciones a pequeña escala, celda a celda.
 * @param zoneNoiseFrequency       Frecuencia del ruido para características a gran escala (zonas de rápidos, remansos).
 * @param totalLength              Longitud total del río en metros.
 * @param spatialResolution        Resolución espacial (dx) en metros.
 * @param initialElevation         Altitud inicial del río en metros.
 * @param concavityFactor          Controla la variación de la pendiente del río en función de la posición.
 * @param averageSlope             Pendiente media del río (adimensional, ej: 0.001 para 1m de caída cada km).
 * @param slopeVariability         Factor de variabilidad para la pendiente (adimensional).
 * @param baseWidth                Ancho base del fondo del río en metros.
 * @param widthVariability         Variación máxima del ancho en metros (ej: 5.0 para +/- 5m).
 * @param baseSideSlope            Pendiente base de los taludes (adimensional).
 * @param sideSlopeVariability     Variación máxima de la pendiente de los taludes.
 * @param slopeSensitivityExponent Factor de sensibilidad geométrica. Un valor de 0.5 implica una relación de raíz
 *                                 cuadrada (típica en hidráulica). Valores mayores exageran el efecto
 *                                 (estrechamientos muy agresivos).
 *                                 - Geometría Hidráulica de Leopold & Maddock (1953)
 * @param roughnessSensitivity     Sensibilidad de la rugosidad a los cambios de pendiente. Basado libremente en
 *                                 la ecuación de Jarrett (n ~ S^0.38) para arroyos de montaña.
 * @param baseManning              Coeficiente de Manning base.
 * @param manningVariability       Variación máxima del coeficiente de Manning.
 * @param baseDecayRateAt20C       Coeficiente de reacción/descomposición base (k) a 20°C, en unidades de s⁻¹.
 * @param decayRateVariability     Variación máxima del coeficiente de reacción.
 * @param decayTurbulenceSensitivity Sensibilidad del Decay a la rugosidad. Un valor de 1.0 implica una relación lineal.
 *                                   0.8 suaviza un poco el efecto para que no sea tan drástico.
 * @param riverPhaseShiftHours     Desfase del pico de PH. Por defecto por la tarde.
 * @param dailyBaseTemperature          Temperatura media diaria del agua en grados Celsius (°C).
 * @param dailyTempVariation       Amplitud de la variación diaria de temperatura en °C (ej: 3.0 para +/- 3°C).
 * @param seasonalTempVariation    Amplitud máxima de la variación anual de temperatura.
 * @param averageAnualTemperature Media de temperatura anual.
 * @param basePh                   pH base del agua (ej: 7.5).
 * @param phVariability            Variación máxima del pH a lo largo del río.
 * @param maxHeadwaterCoolingEffect Enfriamiento máximo en la cabecera del río (nacimiento) en °C.
 * @param headwaterCoolingDistance Distancia en metros sobre la cual el efecto de enfriamiento de la cabecera se disipa.
 * @param widthHeatingFactor       Factor que determina cuánto se calienta el agua en tramos anchos y llanos.
 * @param slopeCoolingFactor       Factor que determina cuánto se enfría el agua en tramos de alta pendiente (rápidos).
 * @param temperatureNoiseAmplitude Amplitud de la variación aleatoria de la temperatura en °C para simular efectos locales.
 * @param originLatitude            Latitud del nacimiento (Punto 0,0 local)
 * @param originLongitud            Longitud del nacimiento
 * @param azimuthAngle              Rotación en grados respecto al Norte geográfico
 */
@Builder
@With
@JsonView(SensorViews.Internal.class)
public record RiverConfig(
        // --- Parámetros de Generación Procedural ---
        long seed,
        float noiseFrequency,
        float detailNoiseFrequency,
        float zoneNoiseFrequency,

        // --- Parámetros Geométricos ---
        float totalLength,
        float spatialResolution,
        float initialElevation,
        float concavityFactor,
        float averageSlope,
        float slopeVariability,
        float baseWidth,
        float widthVariability,
        float baseSideSlope,
        float sideSlopeVariability,

        float slopeSensitivityExponent,
        float roughnessSensitivity,

        // --- Parámetros Hidráulicos ---
        float baseManning,
        float manningVariability,

        // --- Parámetros de Reacción ---
        float baseDecayRateAt20C,
        float decayRateVariability,
        float decayTurbulenceSensitivity,
        float riverPhaseShiftHours,

        // --- Parámetro de dispersión ---
        float baseDispersionAlpha,
        float alphaVariability,

        // --- Parámetros de Calidad de Agua (Temporales) ---
        float dailyBaseTemperature,
        float dailyTempVariation,
        float seasonalTempVariation,
        float averageAnualTemperature,
        float basePh,
        float phVariability,

        // --- Parámetros de Modelo de Temperatura Espacial ---
        float maxHeadwaterCoolingEffect,
        float headwaterCoolingDistance,
        float widthHeatingFactor,
        float slopeCoolingFactor,
        float temperatureNoiseAmplitude,

        // --- Parámetros de localización geoespacial ---
        double originLatitude,
        double originLongitud,
        double azimuthAngle

) {
    // El 'record' se encarga de los getters, constructor, etc.
    public static RiverConfig getTestingRiver(){
        return RiverConfig.builder()
                .seed(12345L)
                .noiseFrequency(0.0f)
                .detailNoiseFrequency(0.05f)
                .zoneNoiseFrequency(0.001f)
                .totalLength(100000)
                .spatialResolution(50)
                .initialElevation(200)
                .concavityFactor(0.4F)
                .averageSlope(0.0002F)
                .slopeVariability(0.0001F)
                .baseWidth(150)
                .widthVariability(40)
                .baseSideSlope(4)
                .sideSlopeVariability(1.5F)
                .slopeSensitivityExponent(0.4f)
                .roughnessSensitivity(0.35f)
                .baseManning(0.030F)
                .manningVariability(0.005F)
                .baseDecayRateAt20C(0.1F)
                .decayRateVariability(0.05F)
                .decayTurbulenceSensitivity(0.8f)
                .riverPhaseShiftHours(15f)
                .baseDispersionAlpha(10)
                .alphaVariability(2)
                .dailyBaseTemperature(15)
                .dailyTempVariation(2)
                .seasonalTempVariation(8)
                .averageAnualTemperature(14)
                .basePh(7.5F)
                .phVariability(0.5F)
                .maxHeadwaterCoolingEffect(4)
                .headwaterCoolingDistance(20000)
                .widthHeatingFactor(1.5F)
                .slopeCoolingFactor(1.0F)
                .temperatureNoiseAmplitude(0.25F)
                .originLatitude(0)
                .originLongitud(0)
                .azimuthAngle(0)
                .build();
    }
}