package projectstalker.ui.viewmodel;

import javafx.beans.property.*;
import org.springframework.stereotype.Component;
import projectstalker.config.RiverConfig;
import projectstalker.ui.view.util.RiverPresets;

@Component
public class RiverEditorViewModel {

    // --- Constantes de lógica de negocio ---
    private static final double NOISE_SLIDER_FACTOR = 2000.0;
    private static final double UI_ROUNDING_FACTOR = 0.1;

    // --- Metadatos ---
    public final StringProperty name = new SimpleStringProperty("");
    public final StringProperty description = new SimpleStringProperty("");

    // --- Geometría ---
    public final DoubleProperty totalLength = new SimpleDoubleProperty();
    public final DoubleProperty baseWidth = new SimpleDoubleProperty();
    public final DoubleProperty widthVarPercent = new SimpleDoubleProperty(); // 0-100%
    public final DoubleProperty slope = new SimpleDoubleProperty();
    public final DoubleProperty slopeVarPercent = new SimpleDoubleProperty(); // 0-100%

    // --- Hidráulica ---
    public final DoubleProperty manning = new SimpleDoubleProperty();
    public final DoubleProperty manningVarPercent = new SimpleDoubleProperty(); // 0-100%

    // --- Procedural (Ruido) ---
    public final IntegerProperty seed = new SimpleIntegerProperty();
    public final DoubleProperty noiseSliderValue = new SimpleDoubleProperty(); // Escala UI (0-100)
    public final DoubleProperty detailFreq = new SimpleDoubleProperty();
    public final DoubleProperty zoneFreq = new SimpleDoubleProperty();

    // --- Físico-Química ---
    public final DoubleProperty dailyBaseTemp = new SimpleDoubleProperty();
    public final DoubleProperty dailyTempVarPercent = new SimpleDoubleProperty(); // 0-100%
    public final DoubleProperty annualBaseTemp = new SimpleDoubleProperty();
    public final DoubleProperty annualTempVarPercent = new SimpleDoubleProperty(); // 0-100%
    public final DoubleProperty basePh = new SimpleDoubleProperty();
    public final DoubleProperty phVarPercent = new SimpleDoubleProperty(); // 0-100%
    public final DoubleProperty dispersion = new SimpleDoubleProperty();

    // --- Avanzados ---
    public final DoubleProperty concavity = new SimpleDoubleProperty();
    public final DoubleProperty sideSlope = new SimpleDoubleProperty();
    public final DoubleProperty slopeSens = new SimpleDoubleProperty();
    public final DoubleProperty decayRate = new SimpleDoubleProperty();
    public final DoubleProperty turbSens = new SimpleDoubleProperty();
    public final DoubleProperty headwaterCooling = new SimpleDoubleProperty();
    public final DoubleProperty widthHeating = new SimpleDoubleProperty();

    public RiverEditorViewModel() {
        // Inicializar con valores por defecto
        loadFromConfig(RiverPresets.standard());
    }

    /**
     * Convierte el estado de la UI (Properties) al DTO de dominio inmutable.
     * Realiza las conversiones matemáticas de UI (%) a Dominio (Absoluto).
     */
    public RiverConfig toDomainConfig() {
        // Cálculos de variabilidad (Porcentaje UI -> Valor Absoluto Config)
        float widthVar = calculateAbsolute(widthVarPercent.get(), baseWidth.get());
        float slopeVar = calculateAbsolute(slopeVarPercent.get(), slope.get());
        float manningVar = calculateAbsolute(manningVarPercent.get(), manning.get());

        float dailyTempVar = calculateAbsolute(dailyTempVarPercent.get(), dailyBaseTemp.get());
        float seasonalTempVar = calculateAbsolute(annualTempVarPercent.get(), annualBaseTemp.get());
        float phVar = calculateAbsolute(phVarPercent.get(), basePh.get());

        // Conversión Frecuencia Ruido
        float mainFreq = (float) (noiseSliderValue.get() / NOISE_SLIDER_FACTOR);

        RiverConfig base = RiverPresets.standard();

        return base
                .withSeed(seed.get())
                // Geometría
                .withTotalLength((float) totalLength.get())
                .withBaseWidth((float) baseWidth.get())
                .withWidthVariability(widthVar)
                .withAverageSlope((float) slope.get())
                .withSlopeVariability(slopeVar)
                // Hidráulica
                .withBaseManning((float) manning.get())
                .withManningVariability(manningVar)
                // Ruido
                .withNoiseFrequency(mainFreq)
                .withDetailNoiseFrequency((float) detailFreq.get())
                .withZoneNoiseFrequency((float) zoneFreq.get())
                // Físico-Química
                .withDailyBaseTemperature((float) dailyBaseTemp.get())
                .withDailyTempVariation(dailyTempVar)
                .withAverageAnualTemperature((float) annualBaseTemp.get())
                .withSeasonalTempVariation(seasonalTempVar)
                .withBasePh((float) basePh.get())
                .withPhVariability(phVar)
                .withBaseDispersionAlpha((float) dispersion.get())
                // Avanzados
                .withConcavityFactor((float) concavity.get())
                .withBaseSideSlope((float) sideSlope.get())
                .withSlopeSensitivityExponent((float) slopeSens.get())
                .withBaseDecayRateAt20C((float) decayRate.get())
                .withDecayTurbulenceSensitivity((float) turbSens.get())
                .withMaxHeadwaterCoolingEffect((float) headwaterCooling.get())
                .withWidthHeatingFactor((float) widthHeating.get());
    }

    /**
     * Carga un DTO de dominio en las Properties de la UI.
     * Realiza las conversiones inversas de Dominio (Absoluto) a UI (%).
     */
    public void loadFromConfig(RiverConfig config) {
        // Geometría
        totalLength.set(config.totalLength());
        baseWidth.set(config.baseWidth());
        widthVarPercent.set(calculatePercentage(config.widthVariability(), config.baseWidth()));

        slope.set(config.averageSlope());
        slopeVarPercent.set(calculatePercentage(config.slopeVariability(), config.averageSlope()));

        // Hidráulica
        manning.set(config.baseManning());
        manningVarPercent.set(calculatePercentage(config.manningVariability(), config.baseManning()));

        // Ruido
        seed.set((int) config.seed());
        double sliderVal = config.noiseFrequency() * NOISE_SLIDER_FACTOR;
        noiseSliderValue.set(Math.min(sliderVal, 100.0)); // Clamp visual
        detailFreq.set(config.detailNoiseFrequency());
        zoneFreq.set(config.zoneNoiseFrequency());

        // Físico-Química
        dailyBaseTemp.set(config.dailyBaseTemperature());
        dailyTempVarPercent.set(calculatePercentage(config.dailyTempVariation(), config.dailyBaseTemperature()));

        annualBaseTemp.set(config.averageAnualTemperature());
        annualTempVarPercent.set(calculatePercentage(config.seasonalTempVariation(), config.averageAnualTemperature()));

        basePh.set(config.basePh());
        phVarPercent.set(calculatePercentage(config.phVariability(), config.basePh()));

        dispersion.set(config.baseDispersionAlpha());

        // Avanzados
        concavity.set(config.concavityFactor());
        sideSlope.set(config.baseSideSlope());
        slopeSens.set(config.slopeSensitivityExponent());
        decayRate.set(config.baseDecayRateAt20C());
        turbSens.set(config.decayTurbulenceSensitivity());
        headwaterCooling.set(config.headwaterCoolingDistance());
        widthHeating.set(config.widthHeatingFactor());
    }

    // --- Helpers Matemáticos ---

    private float calculateAbsolute(double percentage, double baseValue) {
        return (float) ((percentage / 100.0) * baseValue);
    }

    private double calculatePercentage(float variation, float base) {
        if (base == 0) return 0;
        double ratio = (double) variation / base;
        // Lógica de redondeo para que el Spinner muestre valores limpios (ej: 10.0 en vez de 9.9999)
        return UI_ROUNDING_FACTOR * Math.round((ratio * 100) / UI_ROUNDING_FACTOR);
    }
}