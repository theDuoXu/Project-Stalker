package projectstalker.ui.view;

import javafx.application.Platform;
import javafx.beans.value.ObservableValue;
import javafx.collections.FXCollections;
import javafx.fxml.FXML;
import javafx.scene.canvas.Canvas;
import javafx.scene.chart.LineChart;
import javafx.scene.control.*;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.ApplicationEventPublisher;
import org.springframework.stereotype.Component;
import projectstalker.config.RiverConfig;
import projectstalker.domain.dto.twin.TwinCreateRequest;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.factory.RiverGeometryFactory;
import projectstalker.ui.event.RestoreMainViewEvent;
import projectstalker.ui.event.SidebarVisibilityEvent;
import projectstalker.ui.event.TransitoryStatusUpdateEvent;
import projectstalker.ui.renderer.NoiseSignatureRenderer;
import projectstalker.ui.renderer.RiverRenderer;
import projectstalker.ui.service.DigitalTwinClientService;
import projectstalker.ui.service.SimulationEngine;
import projectstalker.ui.view.delegate.RiverEditorCanvasInteractorDelegate;
import projectstalker.ui.view.delegate.RiverUpdateDelegate;
import projectstalker.ui.view.delegate.SimulationControlDelegate;
import projectstalker.ui.view.util.RiverPresets;
import projectstalker.ui.view.util.RiverUiFactory;
import projectstalker.ui.viewmodel.StatusType;
import projectstalker.ui.viewmodel.StatusViewModel;

import java.util.Collections;

@Slf4j
@Component
public class RiverEditorController {

    private final DigitalTwinClientService twinService;
    private final ApplicationEventPublisher eventPublisher;

    // Lógica compleja de dibujo de ríos
    private RiverRenderer renderer; // Instancia del renderer
    private RiverRenderer.RenderMode currentRenderMode = RiverRenderer.RenderMode.MORPHOLOGY;
    private RiverGeometry currentGeometry; // Cache de la geometría actual
    private SimulationEngine simEngine;

    // Para control de cambios
    private RiverConfig initialConfigState;
    private String initialNameState = "";
    private String initialDescState = "";
    private SimulationControlDelegate simDelegate;
    private final RiverEditorCanvasInteractorDelegate canvasInteractor;
    private RiverUpdateDelegate updateDelegate;

    // --- UI: Controles Principales ---
    @FXML
    public TextField nameField;
    @FXML
    public TextArea descField;
    @FXML
    public ComboBox<String> presetCombo;

    // --- UI: Geometría ---
    @FXML
    public Spinner<Double> totalLengthSpinner;
    @FXML
    public Spinner<Double> baseWidthSpinner;
    @FXML
    public Spinner<Double> varWidthSpinner;
    @FXML
    public Spinner<Double> slopeSpinner;
    @FXML
    public Spinner<Double> varSlopeSpinner;

    // --- UI: Panel de simulación ---
    @FXML
    public Label simulationTimeLabel;
    @FXML
    public Label simulationSpeedLabel;

    // --- Hidráulica ---
    @FXML
    public Spinner<Double> manningSpinner;
    @FXML
    public Spinner<Double> varManningSpinner;
    @FXML
    public Label manningDescLabel;

    // --- Procedural (Ruido) ---
    @FXML
    public Spinner<Integer> seedSpinner;
    @FXML
    public Slider noiseSlider;       // Controla 'noiseFrequency' principal
    @FXML
    public Label noiseFreqValueLabel; // Muestra el valor float exacto

    // Inputs avanzados de ruido
    @FXML
    public Spinner<Double> detailFreqSpinner;
    @FXML
    public Spinner<Double> zoneFreqSpinner;

    // --- Físico-Química ---
    @FXML
    public Spinner<Double> dailyBaseTempSpinner;
    @FXML
    public Spinner<Double> anualBaseTempSpinner;
    @FXML
    public Spinner<Double> basePhSpinner;
    @FXML
    public Spinner<Double> dispersionSpinner;
    @FXML
    public Spinner<Double> varDailyBaseTempSpinner;
    @FXML
    public Spinner<Double> varAnualBaseTempSpinner;
    @FXML
    public Spinner<Double> varBasePhSpinner;

    // -- Panel avanzado ---
    @FXML
    public Spinner<Double> concavitySpinner;
    @FXML
    public Spinner<Double> sideSlopeSpinner;
    @FXML
    public Spinner<Double> slopeSensSpinner;
    @FXML
    public Spinner<Double> decayRateSpinner;
    @FXML
    public Spinner<Double> turbSensSpinner;
    @FXML
    public Spinner<Double> headwaterCoolingSpinner;
    @FXML
    public Spinner<Double> widthHeatingSpinner;

    // --- Panel Derecho (Tabs) ---
    @FXML
    public TabPane previewTabs;
    @FXML
    public Canvas morphologyCanvas;
    @FXML
    public Canvas hydrologyCanvas;
    @FXML
    public Canvas noiseCanvas;
    @FXML
    public LineChart<Number, Number> previewChart;
    @FXML
    public Tab morphologyTab;
    @FXML
    public Tab noiseTab;
    @FXML
    public Tab hydrologyTab;
    @FXML
    public Button saveButton;
    @FXML
    public ToggleButton morphologySwitch;

    public RiverEditorController(DigitalTwinClientService twinService, ApplicationEventPublisher eventPublisher, SimulationEngine simEngine, RiverEditorCanvasInteractorDelegate canvasInteractor) {
        this.twinService = twinService;
        this.eventPublisher = eventPublisher;
        this.simEngine = simEngine;
        this.canvasInteractor = canvasInteractor;
    }

    @FXML
    public void initialize() {
        this.renderer = new RiverRenderer(morphologyCanvas);
        this.simDelegate = new SimulationControlDelegate(
                this.simEngine,
                this.eventPublisher,
                this.simulationTimeLabel,
                this.simulationSpeedLabel
        );
        this.updateDelegate = new RiverUpdateDelegate(
                morphologyTab, noiseTab, hydrologyTab,
                this::drawRiverPreview,
                this::drawNoiseHeartBeat,
                this::drawHydrologyTab
        );
        bindUpdateListeners();
        setupTabSelectionClearing();

        setupMorphologySwitch();
        setupPresets();
        setupGeometrySpinners();
        setupManningSpinner();
        setupAdvancedSpinners();
        setupPhysicoChemical();
        setupNoiseSpinners();
        // Cargar default al inicio
        RiverConfig standard = RiverPresets.standard();
        loadConfigToUI(standard);
        saveStateAsInitial(standard, "", "");

        setupCanvas();

        startSimulationEngine();
    }
    private void bindUpdateListeners() {
        // 1. Controles que afectan GEOMETRÍA (Morfología)
        updateDelegate.trackMorphologyChanges(
                totalLengthSpinner.valueProperty(),
                baseWidthSpinner.valueProperty(),
                varWidthSpinner.valueProperty(),
                slopeSpinner.valueProperty(),
                varSlopeSpinner.valueProperty(),
                manningSpinner.valueProperty(),
                varManningSpinner.valueProperty(),
                concavitySpinner.valueProperty(),
                sideSlopeSpinner.valueProperty(),
                slopeSensSpinner.valueProperty(),
                basePhSpinner.valueProperty(),
                varBasePhSpinner.valueProperty(),
                decayRateSpinner.valueProperty(),
                turbSensSpinner.valueProperty()
        );

        // 2. Controles que afectan RUIDO + GEOMETRÍA (Shared)
        updateDelegate.trackSharedChanges(
                seedSpinner.valueProperty(),
                noiseSlider.valueProperty(),
                detailFreqSpinner.valueProperty(),
                zoneFreqSpinner.valueProperty()
        );

        // 3. Controles que afectan HIDROLOGÍA (Física/Química)
        updateDelegate.trackHydrologyChanges(
                dailyBaseTempSpinner.valueProperty(),
                anualBaseTempSpinner.valueProperty(),
                varDailyBaseTempSpinner.valueProperty(),
                varAnualBaseTempSpinner.valueProperty(),
                basePhSpinner.valueProperty(),
                varBasePhSpinner.valueProperty(),
                dispersionSpinner.valueProperty(),
                decayRateSpinner.valueProperty(),
                headwaterCoolingSpinner.valueProperty(),
                widthHeatingSpinner.valueProperty()
        );
    }
    private void setupTabSelectionClearing() {
        morphologyTab.setOnSelectionChanged(e -> {
            if (morphologyTab.isSelected()) {
                updateDelegate.cleanTab(morphologyTab);
                drawRiverPreview();
            }
        });

        noiseTab.setOnSelectionChanged(e -> {
            if (noiseTab.isSelected()) {
                updateDelegate.cleanTab(noiseTab);
                drawNoiseHeartBeat();
            }
        });

        hydrologyTab.setOnSelectionChanged(e -> {
            if (hydrologyTab.isSelected()) {
                updateDelegate.cleanTab(hydrologyTab);
                // Regeneramos la geometría para asegurar consistencia al cambiar de tab
                this.currentGeometry = RiverGeometryFactory.createRealisticRiver(buildConfigFromUI());
                drawHydrologyTab();
            }
        });
    }

    private void setupMorphologySwitch() {
        morphologySwitch.selectedProperty().addListener((observable, oldValue, newValue) -> {
            onMorphologyCanvasModeChange(newValue);
        });
    }

    private void startSimulationEngine() {
        simEngine.setOnFrameReady(snapshot -> {
            // Pintar en el Canvas de Hidrología (esto se mantiene aquí por ahora)
            renderer.renderHydrology(
                    hydrologyCanvas,
                    currentGeometry,
                    snapshot,
                    canvasInteractor.getMouseX(),
                    canvasInteractor.getMouseY());

            // Delegar actualización de UI
            simDelegate.updateTime(snapshot.timeSeconds());
        });
        simEngine.start();
    }


    // =========================================================================
    // 1. CONFIGURACIÓN DE CONTROLES (SETUP)
    // =========================================================================

    private void setupCanvas() {
        // El canvas debe redibujarse si cambia el tamaño del contenedor padre
        this.updateDelegate.trackCanvasSizeChanges(noiseCanvas, this::drawNoiseHeartBeat);
        this.updateDelegate.trackCanvasSizeChanges(morphologyCanvas, this::drawNoiseHeartBeat);
        this.updateDelegate.trackCanvasSizeChanges(hydrologyCanvas, this::drawNoiseHeartBeat);

        canvasInteractor.bind(
                this::redrawMorphology,     // Qué hacer al mover el ratón (Repaint)
                renderer::reloadThemeColors, // Qué hacer al cargar la escena (Theme)
                morphologyCanvas, hydrologyCanvas // Los canvas afectados
        );
    }
    private void redrawMorphology() {
        if (currentGeometry != null) {
            renderer.render(
                    currentGeometry,
                    currentRenderMode,
                    canvasInteractor.getMouseX(),
                    canvasInteractor.getMouseY()
            );
        }
    }

    private void setupGeometrySpinners() {
        // Longitud: 1km a 500km, paso de 10m;
        RiverUiFactory.configureSpinner(totalLengthSpinner, 1000, 500000, 50000, 10);
        // Ancho: 5m a 2km, paso de 5m
        RiverUiFactory.configureSpinner(baseWidthSpinner, 5, 2000, 150, 5);
        RiverUiFactory.configureSpinner(varWidthSpinner, 0, 100, 10, 0.1);

        // Pendiente: 0.00001 a 0.1, paso fino de 0.0001
        RiverUiFactory.configureScientificSpinner(slopeSpinner, 0.00001, 0.1, 0.0002, 0.0001, "%.5f");
        RiverUiFactory.configureSpinner(varSlopeSpinner, 0, 100, 10, 0.1);
    }

    private void setupManningSpinner() {
        // Rango científico: 0.010 (Canal liso) a 0.150 (Máximo razonable). Paso: 0.001
        RiverUiFactory.configureScientificSpinner(manningSpinner, 0.010, 0.150, 0.030, 0.001, "%.3f");
        RiverUiFactory.configureSpinner(varManningSpinner, 0, 100, 10, 0.1);

        // Listener para dar feedback textual sobre el tipo de material
        manningSpinner.valueProperty().addListener((obs, oldVal, newVal) -> updateManningLabel(newVal));
    }

    private void setupNoiseSpinners() {
        // 1. Seed (Semilla) - Entero Grande
        // Rango: 0 a 999999999. Paso: 1 o 10000. Usamos Long para la seguridad del tipo.
        RiverUiFactory.configureSpinner(seedSpinner,0, 9999999, 1000, 1 );
        // 2. Frecuencia de Detalle (detailFreqSpinner) - Valor fino
        // Rango: 0.001 a 1.0. Paso: 0.001 (muy fino).
        RiverUiFactory.configureScientificSpinner(detailFreqSpinner, 0.001, 1.0, 0.05, 0.001, "%.4f");

        // 3. Frecuencia Zonal (zoneFreqSpinner) - Valor muy fino (escala macro)
        // Rango: 0.0001 a 0.1. Paso: 0.0001 (aún más fino).
        RiverUiFactory.configureScientificSpinner(zoneFreqSpinner, 0.0001, 0.1, 0.001, 0.0001, "%.5f");
    }

    private void setupAdvancedSpinners() {
        // Concavidad (0.4 default)
        RiverUiFactory.configureSpinner(concavitySpinner, 0.0, 1.0, 0.4, 0.05);

        // Talud (z) (4.0 default)
        RiverUiFactory.configureSpinner(sideSlopeSpinner, 0.1, 10.0, 4.0, 0.1);

        // Sensibilidad Ancho-Pendiente (0.4 default)
        RiverUiFactory.configureSpinner(slopeSensSpinner, 0.1, 1.0, 0.4, 0.01);

        // Tasa Decay k20 (0.1 default)
        RiverUiFactory.configureSpinner(decayRateSpinner, 0.0, 2.0, 0.1, 0.01);

        // Sensibilidad Turbulencia (0.8 default)
        RiverUiFactory.configureSpinner(turbSensSpinner, 0.0, 2.0, 0.8, 0.1);

        // Enfriamiento Cabecera (4.0 default)
        RiverUiFactory.configureSpinner(headwaterCoolingSpinner, 0.0, 15.0, 4.0, 0.5);

        // Factor Calentamiento por Ancho (1.5 default)
        RiverUiFactory.configureSpinner(widthHeatingSpinner, 0.0, 5.0, 1.5, 0.1);
    }

    private void setupPhysicoChemical() {
        // Temperatura Base Diaria
        RiverUiFactory.configureSpinner(dailyBaseTempSpinner, -5.0, 40.0, 15.0, 0.1);
        RiverUiFactory.configureSpinner(varDailyBaseTempSpinner, 0.0, 100.0, 20.0, 0.1);

        // Temperatura Base Anual
        RiverUiFactory.configureSpinner(anualBaseTempSpinner, -5.0, 40.0, 15.0, 0.1);
        RiverUiFactory.configureSpinner(varAnualBaseTempSpinner, 0.0, 100.0, 10.0, 0.1);

        // pH Base
        RiverUiFactory.configureSpinner(basePhSpinner, 0.0, 14.0, 7.5, 0.1);
        RiverUiFactory.configureSpinner(varBasePhSpinner, 0.0, 50.0, 10.0, 0.1);

        // Dispersión
        RiverUiFactory.configureSpinner(dispersionSpinner, 0.1, 100.0, 10.0, 1.0);


//        dispersionSpinner.valueProperty().addListener(this::onAnySpinnerUpdate) todavía no se usa pero no borrar

    }

    private void setupPresets() {
        presetCombo.setItems(FXCollections.observableArrayList("Río Estándar (Tramo Medio)", "Torrente de Alta Montaña", "Llanura / Delta Ancho"));
        presetCombo.getSelectionModel().select(0);
        presetCombo.setOnAction(e -> applyPreset(presetCombo.getSelectionModel().getSelectedIndex()));
    }

    // =========================================================================
    // 2. LÓGICA VISUAL (CANVAS & LABELS)
    // =========================================================================

    private void updateManningLabel(Double n) {
        String text;
        if (n < 0.015) text = "Hormigón / Metal liso";
        else if (n < 0.025) text = "Tierra limpia / Arena";
        else if (n < 0.035) text = "Grava / Lecho natural estándar";
        else if (n < 0.050) text = "Rocas / Montaña / Algo de vegetación";
        else if (n < 0.075) text = "Vegetación densa / Troncos";
        else text = "Obstrucción extrema / Manglares";

        manningDescLabel.setText(text);

        if (n > 0.1) manningDescLabel.setStyle("-fx-text-fill: -color-danger-fg; -fx-font-size: 10px;");
        else manningDescLabel.setStyle("-fx-text-fill: -color-fg-muted; -fx-font-size: 10px;");
    }

    /**
     * Dibuja una vista técnica combinada:
     * 1. Planta (Top-down): Ancho del cauce y zonas de fricción.
     * 2. Perfil (Side-view): Caída de elevación a lo largo del río.
     */
    private void drawRiverPreview() {
        RiverConfig config = buildConfigFromUI();
        this.currentGeometry = RiverGeometryFactory.createRealisticRiver(config);
        drawHydrologyTab();
    }

    private void drawHydrologyTab() {
        simEngine.loadRiver(currentGeometry, buildConfigFromUI());
        // Delegar pintado al Renderer
        renderer.render(currentGeometry, currentRenderMode, canvasInteractor.getMouseX(), canvasInteractor.getMouseY());
    }

    /**
     * Dibuja la firma combinada del Ruido Perlin (Heartbeat) usando la misma
     * lógica de FastNoiseLite que la fábrica, para previsualizar la rugosidad.
     */
    private void drawNoiseHeartBeat() {
        if (detailFreqSpinner.getValue() == null || zoneFreqSpinner.getValue() == null) {
            log.warn("Noise Spinners no inicializados. Saltando HeartBeat draw.");
            return;
        }
        NoiseSignatureRenderer.render(noiseCanvas, buildConfigFromUI());
    }

    // =========================================================================
    // 3. MAPEO DE DATOS (DTO <-> UI)
    // =========================================================================

    /**
     * Guarda una "foto" del estado actual para saber si luego ha cambiado.
     */
    private void saveStateAsInitial(RiverConfig config, String name, String desc) {
        this.initialConfigState = config;
        this.initialNameState = name == null ? "" : name;
        this.initialDescState = desc == null ? "" : desc;
    }

    /**
     * Comprueba si lo que hay en pantalla es diferente a lo que había al entrar.
     */
    private boolean hasUnsavedChanges() {
        RiverConfig currentConfig = buildConfigFromUI();
        String currentName = nameField.getText();
        String currentDesc = descField.getText();

        boolean configChanged = !currentConfig.equals(initialConfigState);
        boolean nameChanged = !currentName.equals(initialNameState);
        boolean descChanged = !currentDesc.equals(initialDescState);

        return configChanged || nameChanged || descChanged;
    }


    private void applyPreset(int index) {
        switch (index) {
            case 1 -> loadConfigToUI(RiverPresets.mountainTorrent());
            case 2 -> loadConfigToUI(RiverPresets.widePlains());
            default -> loadConfigToUI(RiverPresets.standard());
        }
    }


    private void loadConfigToUI(RiverConfig config) {
        final double ROUNDING_FACTOR = 0.1;

        // --- GEOMETRÍA ---
        totalLengthSpinner.getValueFactory().setValue((double) config.totalLength());
        baseWidthSpinner.getValueFactory().setValue((double) config.baseWidth());

        // Proporción de variación de ancho: se aplica el redondeo
        double widthRatio = (double) config.widthVariability() / config.baseWidth();
        varWidthSpinner.getValueFactory().setValue(ROUNDING_FACTOR * Math.round(widthRatio / ROUNDING_FACTOR) * 100);


        // --- PENDIENTE ---
        slopeSpinner.getValueFactory().setValue((double) config.averageSlope());

        // Proporción de variación de pendiente: se aplica el redondeo
        double slopeRatio = (double) config.slopeVariability() / config.averageSlope();
        varSlopeSpinner.getValueFactory().setValue(ROUNDING_FACTOR * Math.round(slopeRatio / ROUNDING_FACTOR) * 100);


        // --- RUGOSIDAD (MANNING) ---
        manningSpinner.getValueFactory().setValue((double) config.baseManning());
        updateManningLabel((double) config.baseManning());

        // Proporción de variación de Manning: se aplica el redondeo
        double manningRatio = (double) config.manningVariability() / config.baseManning();
        varManningSpinner.getValueFactory().setValue(ROUNDING_FACTOR * Math.round(manningRatio / ROUNDING_FACTOR) * 100);


        // --- SEMILLA DE RUIDO ---
        seedSpinner.getValueFactory().setValue((int) config.seed());


        // --- FRECUENCIA DE RUIDO (SLIDER) ---
        // El slider no usa un ValueFactory de forma directa, se calcula y se establece el valor
        double sliderVal = config.noiseFrequency() * 2000.0;
        if (sliderVal > 100) sliderVal = 100;
        noiseSlider.setValue(sliderVal);
        noiseFreqValueLabel.setText(String.format("%.5f", config.noiseFrequency()));

        // --- DETALLES DE RUIDO ---
        detailFreqSpinner.getValueFactory().setValue((double) config.detailNoiseFrequency());
        zoneFreqSpinner.getValueFactory().setValue((double) config.zoneNoiseFrequency());


        // --- TEMPERATURA ---
        dailyBaseTempSpinner.getValueFactory().setValue((double) config.dailyBaseTemperature());
        anualBaseTempSpinner.getValueFactory().setValue((double) config.averageAnualTemperature());

        // Proporción de amplitud de ruido de temperatura: se aplica el redondeo
        double tempRatio = (double) config.dailyTempVariation() / config.dailyBaseTemperature();
        varDailyBaseTempSpinner.getValueFactory().setValue(ROUNDING_FACTOR * Math.round(tempRatio / ROUNDING_FACTOR) * 100);
        tempRatio = (double) config.seasonalTempVariation() / config.averageAnualTemperature();
        varAnualBaseTempSpinner.getValueFactory().setValue(ROUNDING_FACTOR * Math.round(tempRatio / ROUNDING_FACTOR) * 100);

        // --- pH ---
        basePhSpinner.getValueFactory().setValue((double) config.basePh());

        // Proporción de variabilidad de pH: se aplica el redondeo
        double phRatio = (double) config.phVariability() / config.basePh();
        varBasePhSpinner.getValueFactory().setValue(ROUNDING_FACTOR * Math.round(phRatio / ROUNDING_FACTOR) * 100);


        // --- DISPERSIÓN ---
        dispersionSpinner.getValueFactory().setValue((double) config.baseDispersionAlpha());

        // -- AVANZADOS ---
        concavitySpinner.getValueFactory().setValue((double) config.concavityFactor());
        sideSlopeSpinner.getValueFactory().setValue((double) config.baseSideSlope());
        slopeSensSpinner.getValueFactory().setValue((double) config.slopeSensitivityExponent());
        decayRateSpinner.getValueFactory().setValue((double) config.baseDecayRateAt20C());
        turbSensSpinner.getValueFactory().setValue((double) config.decayTurbulenceSensitivity());
        headwaterCoolingSpinner.getValueFactory().setValue((double) config.headwaterCoolingDistance());
        widthHeatingSpinner.getValueFactory().setValue((double) config.widthHeatingFactor());

        eventPublisher.publishEvent(
                new TransitoryStatusUpdateEvent(
                        "Preconfiguración cargada!",
                        StatusViewModel.TransitionTime.IMMEDIATE,
                        StatusType.SUCCESS));
    }

    private RiverConfig buildConfigFromUI() {
        RiverConfig base = RiverPresets.standard();

        long seed = base.seed();
        try {
            seed = seedSpinner.getValue();
        } catch (NumberFormatException ignored) {
        }

        // Frecuencia real calculada desde el slider
        float mainFreq = (float) (noiseSlider.getValue() / 2000.0);

        return base
                // --- 1. GEOMETRÍA (Base y Variabilidad) ---
                .withSeed(seedSpinner.getValue()).withTotalLength(totalLengthSpinner.getValue().floatValue()).withBaseWidth(baseWidthSpinner.getValue().floatValue()).withWidthVariability((float) (varWidthSpinner.getValue() / 100 * baseWidthSpinner.getValue())) // Cálculo inverso: varWidth = ratio * baseWidth
                .withAverageSlope(slopeSpinner.getValue().floatValue()).withSlopeVariability((float) (varSlopeSpinner.getValue() / 100 * slopeSpinner.getValue())) // Cálculo inverso: varSlope = ratio * avgSlope

                // --- 2. HIDRÁULICA ---
                .withBaseManning(manningSpinner.getValue().floatValue()).withManningVariability((float) (varManningSpinner.getValue() / 100 * manningSpinner.getValue())) // Cálculo inverso: varManning = ratio * baseManning

                // --- 3. PROCEDURAL (Ruido) ---
                .withNoiseFrequency(mainFreq).withDetailNoiseFrequency(detailFreqSpinner.getValue().floatValue()).withZoneNoiseFrequency(zoneFreqSpinner.getValue().floatValue())

                // --- 4. FÍSICO-QUÍMICA (Temperatura, pH, Dispersión) ---
                // Temperatura
                .withDailyBaseTemperature(dailyBaseTempSpinner.getValue().floatValue()).withDailyTempVariation((float) (varDailyBaseTempSpinner.getValue() / 100 * dailyBaseTempSpinner.getValue())).withAverageAnualTemperature(anualBaseTempSpinner.getValue().floatValue()).withSeasonalTempVariation((float) (varAnualBaseTempSpinner.getValue() / 100 * anualBaseTempSpinner.getValue()))

                // pH
                .withBasePh(basePhSpinner.getValue().floatValue()).withPhVariability((float) (varBasePhSpinner.getValue() / 100 * basePhSpinner.getValue())) // Cálculo inverso: varPh = ratio * basePh

                // Dispersión
                .withBaseDispersionAlpha(dispersionSpinner.getValue().floatValue())

                // Avanzados
                .withConcavityFactor(concavitySpinner.getValue().floatValue()).withBaseSideSlope(sideSlopeSpinner.getValue().floatValue()).withSlopeSensitivityExponent(slopeSensSpinner.getValue().floatValue()).withBaseDecayRateAt20C(decayRateSpinner.getValue().floatValue()).withDecayTurbulenceSensitivity(turbSensSpinner.getValue().floatValue()).withMaxHeadwaterCoolingEffect(headwaterCoolingSpinner.getValue().floatValue()).withWidthHeatingFactor(widthHeatingSpinner.getValue().floatValue());
    }

    // =========================================================================
    // 4. ACCIONES (EVENTS)
    // =========================================================================

    @FXML
    public void onSave() {
        if (nameField.getText() == null || nameField.getText().isBlank()) {
            showAlert(Alert.AlertType.WARNING, "Validación", "El nombre del río es obligatorio.");
            return;
        }

        saveButton.setDisable(true);
        saveButton.setText("Guardando...");

        RiverConfig config = buildConfigFromUI();
        var request = new TwinCreateRequest(nameField.getText(), descField.getText(), config, Collections.emptyList());

        twinService.createTwin(request).subscribe(summary -> Platform.runLater(() -> {
            saveButton.setDisable(false);
            saveButton.setText("Crear Gemelo");
            showAlert(Alert.AlertType.INFORMATION, "Éxito", "Gemelo Digital '" + summary.name() + "' creado correctamente.");

            // Eventos de lista
            eventPublisher.publishEvent(new SidebarVisibilityEvent(true));

            // Recuperar vista inicial
            eventPublisher.publishEvent(new RestoreMainViewEvent());

            // Notificar éxito
            eventPublisher.publishEvent(
                    new TransitoryStatusUpdateEvent(
                            "Configuración guardada con éxito",
                            StatusViewModel.TransitionTime.SHORT,
                            StatusType.SUCCESS));

        }), error -> Platform.runLater(() -> {
            saveButton.setDisable(false);
            saveButton.setText("Crear Gemelo");
            showAlert(Alert.AlertType.ERROR, "Error", "Fallo al guardar: " + error.getMessage());
        }));
    }

    @FXML
    public void onCancel() {
        if (hasUnsavedChanges()) {
            Alert alert = new Alert(Alert.AlertType.CONFIRMATION);
            alert.setTitle("Cambios sin guardar");
            alert.setHeaderText("¿Estás seguro de que quieres salir?");
            alert.setContentText("Perderás la configuración actual del río.");

            // Si el usuario dice Cancelar en el diálogo, nos quedamos aquí.
            if (alert.showAndWait().orElse(ButtonType.CANCEL) == ButtonType.CANCEL) {
                return;
            }
            // Eventos de lista
            eventPublisher.publishEvent(new SidebarVisibilityEvent(true));

            // Recuperar vista inicial
            eventPublisher.publishEvent(new RestoreMainViewEvent());
        }

        loadConfigToUI(RiverPresets.standard());
        nameField.clear();
        descField.clear();
        previewChart.getData().clear();
        drawRiverPreview();

        // Restaurar sidebar
        eventPublisher.publishEvent(new SidebarVisibilityEvent(true));

        // Recuperar vista inicial
        eventPublisher.publishEvent(new RestoreMainViewEvent());
    }

    @FXML
    public void onSimRestart() {
        simDelegate.restart();
    }

    @FXML
    public void onSimRewind() {
        simDelegate.rewind();
    }

    @FXML
    public void onSimPause() {
        simDelegate.pause();
    }

    @FXML
    public void onSimPlay() {
        simDelegate.play();
    }

    @FXML
    public void onSimAccelerate() {
        simDelegate.accelerate();
    }

    @FXML
    private void onMorphologyCanvasModeChange(Boolean newValue) {
        currentRenderMode = RiverRenderer.RenderMode.fromBoolean(newValue);
    }

    // =========================================================================
    // 5. HELPERS
    // =========================================================================

    private void showAlert(Alert.AlertType type, String title, String content) {
        Alert alert = new Alert(type);
        alert.setTitle(title);
        alert.setHeaderText(null);
        alert.setContentText(content);
        alert.showAndWait();
    }
}