package projectstalker.ui.view;

import javafx.application.Platform;
import javafx.beans.property.DoubleProperty;
import javafx.beans.property.IntegerProperty;
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
import projectstalker.domain.dto.twin.TwinDetailDTO;
import projectstalker.domain.dto.twin.TwinSummaryDTO;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.factory.RiverGeometryFactory;
import projectstalker.ui.event.RestoreMainViewEvent;
import projectstalker.ui.event.SidebarVisibilityEvent;
import projectstalker.ui.event.TransitoryStatusUpdateEvent;
import projectstalker.ui.event.TwinListRefreshEvent;
import projectstalker.ui.renderer.NoiseSignatureRenderer;
import projectstalker.ui.renderer.RiverRenderer;
import projectstalker.ui.service.DigitalTwinClientService;
import projectstalker.ui.service.SimulationEngine;
import projectstalker.ui.view.delegate.RiverEditorCanvasInteractorDelegate;
import projectstalker.ui.view.delegate.RiverUpdateDelegate;
import projectstalker.ui.view.delegate.SimulationControlDelegate;
import projectstalker.ui.view.util.RiverPresets;
import projectstalker.ui.view.util.RiverUiFactory;
import projectstalker.ui.viewmodel.RiverEditorViewModel;
import projectstalker.ui.viewmodel.StatusType;
import projectstalker.ui.viewmodel.StatusViewModel;

import java.time.Duration;
import java.time.temporal.ChronoUnit;
import java.util.Collections;

@Slf4j
@Component
public class RiverEditorController {
    private final Duration TIMEOUT_DURATION = Duration.of(60, ChronoUnit.SECONDS);
    private final DigitalTwinClientService twinService;
    private final ApplicationEventPublisher eventPublisher;

    // Lógica compleja de dibujo de ríos
    private RiverRenderer renderer; // Instancia del renderer
    private RiverRenderer.RenderMode currentRenderMode = RiverRenderer.RenderMode.MORPHOLOGY;
    private RiverGeometry currentGeometry; // Cache de la geometría actual
    private final SimulationEngine simEngine;

    // Para control de cambios
    private RiverConfig initialConfigState;
    private String initialNameState = "";
    private String initialDescState = "";
    private String editingTwinId = null;

    // Delegados y ViewModel
    private SimulationControlDelegate simDelegate;
    private final RiverEditorCanvasInteractorDelegate canvasInteractor;
    private RiverUpdateDelegate updateDelegate;
    private final RiverEditorViewModel viewModel;

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
    public Slider noiseSlider; // Controla 'noiseFrequency' principal
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
    public Tab morphologyTab;
    @FXML
    public Tab noiseTab;
    @FXML
    public Tab hydrologyTab;
    @FXML
    public Button saveButton;
    @FXML
    public ToggleButton morphologySwitch;

    public RiverEditorController(DigitalTwinClientService twinService,
            ApplicationEventPublisher eventPublisher,
            SimulationEngine simEngine,
            RiverEditorCanvasInteractorDelegate canvasInteractor,
            RiverEditorViewModel viewModel) { // Inyección del VM
        this.twinService = twinService;
        this.eventPublisher = eventPublisher;
        this.simEngine = simEngine;
        this.canvasInteractor = canvasInteractor;
        this.viewModel = viewModel;
    }

    @FXML
    public void initialize() {
        this.renderer = new RiverRenderer(morphologyCanvas);
        this.simDelegate = new SimulationControlDelegate(
                this.simEngine,
                this.eventPublisher,
                this.simulationTimeLabel,
                this.simulationSpeedLabel);
        this.updateDelegate = new RiverUpdateDelegate(
                morphologyTab, noiseTab, hydrologyTab,
                this::drawRiverPreview,
                this::drawNoiseHeartBeat,
                this::drawHydrologyTab);

        setupMorphologySwitch();
        setupPresets();

        // Configuración de rangos (UI Factories)
        setupGeometrySpinners();
        setupManningSpinner();
        setupAdvancedSpinners();
        setupPhysicoChemical();
        setupNoiseSpinners();

        // --- FASE 2: BINDING VIEWMODEL ---
        bindViewModel();

        // Configuración de listeners de actualización (Delegate)
        bindUpdateListeners();
        setupTabSelectionClearing();

        // Cargar default al inicio
        RiverConfig standard = RiverPresets.standard();
        loadConfigToUI(standard);
        saveStateAsInitial(standard, "", "");

        setupCanvas();

        startSimulationEngine();
    }

    // =========================================================================
    // FASE 2: VINCULACIÓN DE DATOS (BINDING)
    // =========================================================================
    private void bindViewModel() {
        // Metadatos
        nameField.textProperty().bindBidirectional(viewModel.name);
        descField.textProperty().bindBidirectional(viewModel.description);

        // Geometría
        bindSpinner(totalLengthSpinner, viewModel.totalLength);
        bindSpinner(baseWidthSpinner, viewModel.baseWidth);
        bindSpinner(varWidthSpinner, viewModel.widthVarPercent);
        bindSpinner(slopeSpinner, viewModel.slope);
        bindSpinner(varSlopeSpinner, viewModel.slopeVarPercent);

        // Hidráulica
        bindSpinner(manningSpinner, viewModel.manning);
        bindSpinner(varManningSpinner, viewModel.manningVarPercent);

        // Ruido
        bindSpinner(seedSpinner, viewModel.seed);
        noiseSlider.valueProperty().bindBidirectional(viewModel.noiseSliderValue);
        bindSpinner(detailFreqSpinner, viewModel.detailFreq);
        bindSpinner(zoneFreqSpinner, viewModel.zoneFreq);

        // Físico-Química
        bindSpinner(dailyBaseTempSpinner, viewModel.dailyBaseTemp);
        bindSpinner(varDailyBaseTempSpinner, viewModel.dailyTempVarPercent);
        bindSpinner(anualBaseTempSpinner, viewModel.annualBaseTemp);
        bindSpinner(varAnualBaseTempSpinner, viewModel.annualTempVarPercent);
        bindSpinner(basePhSpinner, viewModel.basePh);
        bindSpinner(varBasePhSpinner, viewModel.phVarPercent);
        bindSpinner(dispersionSpinner, viewModel.dispersion);

        // Avanzados
        bindSpinner(concavitySpinner, viewModel.concavity);
        bindSpinner(sideSlopeSpinner, viewModel.sideSlope);
        bindSpinner(slopeSensSpinner, viewModel.slopeSens);
        bindSpinner(decayRateSpinner, viewModel.decayRate);
        bindSpinner(turbSensSpinner, viewModel.turbSens);
        bindSpinner(headwaterCoolingSpinner, viewModel.headwaterCooling);
        bindSpinner(widthHeatingSpinner, viewModel.widthHeating);
    }

    // Helpers de binding
    private void bindSpinner(Spinner<Double> spinner, DoubleProperty property) {
        SpinnerValueFactory<Double> factory = spinner.getValueFactory();

        // 1. Sincronización Inicial (Para no quedarse en default)
        if (factory.getValue() != null)
            property.set(factory.getValue());
        else
            factory.setValue(property.get());

        // 2. Listener: ViewModel -> Spinner (Cuando cargamos config)
        property.addListener((obs, oldVal, newVal) -> {
            if (newVal != null && !newVal.equals(factory.getValue())) {
                factory.setValue(newVal.doubleValue());
            }
        });

        // 3. Listener: Spinner -> ViewModel (Cuando el usuario edita)
        factory.valueProperty().addListener((obs, oldVal, newVal) -> {
            if (newVal != null && newVal != property.get()) {
                property.set(newVal);
            }
        });
    }

    private void bindSpinner(Spinner<Integer> spinner, IntegerProperty property) {
        SpinnerValueFactory<Integer> factory = spinner.getValueFactory();

        // 1. Sincronización Inicial
        if (factory.getValue() != null)
            property.set(factory.getValue());
        else
            factory.setValue(property.get());

        // 2. Listener: ViewModel -> Spinner
        property.addListener((obs, oldVal, newVal) -> {
            if (newVal != null && !newVal.equals(factory.getValue())) {
                factory.setValue(newVal.intValue());
            }
        });

        // 3. Listener: Spinner -> ViewModel
        factory.valueProperty().addListener((obs, oldVal, newVal) -> {
            if (newVal != null && newVal != property.get()) {
                property.set(newVal);
            }
        });
    }

    // =========================================================================
    // CONFIGURACIÓN DE LISTENERS (UPDATE DELEGATE)
    // =========================================================================

    private void bindUpdateListeners() {
        // 1. Controles que afectan GEOMETRÍA (Morfología)
        updateDelegate.trackMorphologyChanges(
                viewModel.totalLength,
                viewModel.baseWidth,
                viewModel.widthVarPercent,
                viewModel.slope,
                viewModel.slopeVarPercent,
                viewModel.manning,
                viewModel.manningVarPercent,
                viewModel.concavity,
                viewModel.sideSlope,
                viewModel.slopeSens,
                viewModel.basePh,
                viewModel.phVarPercent,
                viewModel.decayRate,
                viewModel.turbSens);

        // 2. Controles que afectan RUIDO + GEOMETRÍA + HYDROLOGY (Shared)
        updateDelegate.trackSharedChanges(
                viewModel.seed,
                viewModel.noiseSliderValue,
                viewModel.detailFreq,
                viewModel.zoneFreq);

        // 3. Controles que afectan HIDROLOGÍA (Física/Química)
        updateDelegate.trackHydrologyChanges(
                viewModel.dailyBaseTemp,
                viewModel.annualBaseTemp,
                viewModel.dailyTempVarPercent,
                viewModel.annualTempVarPercent,
                viewModel.basePh,
                viewModel.phVarPercent,
                viewModel.dispersion,
                viewModel.decayRate,
                viewModel.headwaterCooling,
                viewModel.widthHeating,
                viewModel.noiseSliderValue);
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
    // 1. CONFIGURACIÓN DE CONTROLES (SETUP - Solo Rangos)
    // =========================================================================

    private void setupCanvas() {
        this.updateDelegate.trackCanvasSizeChanges(noiseCanvas, this::drawNoiseHeartBeat);
        this.updateDelegate.trackCanvasSizeChanges(morphologyCanvas, this::drawRiverPreview);
        this.updateDelegate.trackCanvasSizeChanges(hydrologyCanvas, this::drawHydrologyTab);

        canvasInteractor.bind(
                this::redrawMorphology,
                renderer::reloadThemeColors,
                morphologyCanvas, hydrologyCanvas);
    }

    private void redrawMorphology() {
        if (currentGeometry != null) {
            renderer.render(
                    currentGeometry,
                    currentRenderMode,
                    canvasInteractor.getMouseX(),
                    canvasInteractor.getMouseY());
        }
    }

    private void setupGeometrySpinners() {
        RiverUiFactory.configureSpinner(totalLengthSpinner, 1000, 500000, 50000, 10);
        RiverUiFactory.configureSpinner(baseWidthSpinner, 5, 2000, 150, 5);
        RiverUiFactory.configureSpinner(varWidthSpinner, 0, 100, 10, 0.1);
        RiverUiFactory.configureScientificSpinner(slopeSpinner, 0.00001, 0.1, 0.0002, 0.0001, "%.5f");
        RiverUiFactory.configureSpinner(varSlopeSpinner, 0, 100, 10, 0.1);
    }

    private void setupManningSpinner() {
        RiverUiFactory.configureScientificSpinner(manningSpinner, 0.010, 0.150, 0.030, 0.001, "%.3f");
        RiverUiFactory.configureSpinner(varManningSpinner, 0, 100, 10, 0.1);
        manningSpinner.valueProperty().addListener((obs, oldVal, newVal) -> updateManningLabel(newVal));
    }

    private void setupNoiseSpinners() {
        RiverUiFactory.configureSpinner(seedSpinner, 0, 9999999, 1000, 1);
        RiverUiFactory.configureScientificSpinner(detailFreqSpinner, 0.001, 1.0, 0.05, 0.001, "%.4f");
        RiverUiFactory.configureScientificSpinner(zoneFreqSpinner, 0.0001, 0.1, 0.001, 0.0001, "%.5f");
        noiseSlider.valueProperty().addListener((obs, oldVal, newVal) -> {
            double freq = newVal.doubleValue() / 2000.0;
            noiseFreqValueLabel.setText(String.format("%.5f", freq));
        });
    }

    private void setupAdvancedSpinners() {
        RiverUiFactory.configureSpinner(concavitySpinner, 0.0, 1.0, 0.4, 0.05);
        RiverUiFactory.configureSpinner(sideSlopeSpinner, 0.1, 10.0, 4.0, 0.1);
        RiverUiFactory.configureSpinner(slopeSensSpinner, 0.1, 1.0, 0.4, 0.01);
        RiverUiFactory.configureSpinner(decayRateSpinner, 0.0, 2.0, 0.1, 0.01);
        RiverUiFactory.configureSpinner(turbSensSpinner, 0.0, 2.0, 0.8, 0.1);
        RiverUiFactory.configureSpinner(headwaterCoolingSpinner, 0.0, 15.0, 4.0, 0.5);
        RiverUiFactory.configureSpinner(widthHeatingSpinner, 0.0, 5.0, 1.5, 0.1);
    }

    private void setupPhysicoChemical() {
        RiverUiFactory.configureSpinner(dailyBaseTempSpinner, -5.0, 40.0, 15.0, 0.1);
        RiverUiFactory.configureSpinner(varDailyBaseTempSpinner, 0.0, 100.0, 20.0, 0.1);
        RiverUiFactory.configureSpinner(anualBaseTempSpinner, -5.0, 40.0, 15.0, 0.1);
        RiverUiFactory.configureSpinner(varAnualBaseTempSpinner, 0.0, 100.0, 10.0, 0.1);
        RiverUiFactory.configureSpinner(basePhSpinner, 0.0, 14.0, 7.5, 0.1);
        RiverUiFactory.configureSpinner(varBasePhSpinner, 0.0, 50.0, 10.0, 0.1);
        RiverUiFactory.configureSpinner(dispersionSpinner, 0.1, 100.0, 10.0, 1.0);
    }

    private void setupPresets() {
        presetCombo.setItems(FXCollections.observableArrayList("Río Estándar (Tramo Medio)", "Torrente de Alta Montaña",
                "Llanura / Delta Ancho"));
        presetCombo.getSelectionModel().select(0);
        presetCombo.setOnAction(e -> applyPreset(presetCombo.getSelectionModel().getSelectedIndex()));
    }

    // =========================================================================
    // 2. LÓGICA VISUAL (CANVAS & LABELS)
    // =========================================================================
    public void setEditingTwin(TwinSummaryDTO twin) {
        // 1. Inicialización básica de UI
        this.editingTwinId = twin.id();
        this.nameField.setText(twin.name());
        this.descField.setText(twin.description());
        this.saveButton.setText("Guardar Cambios");

        // 2. Recuperar configuración completa del Backend
        twinService.getTwinDetails(twin.id())
                .subscribe(
                        detail -> Platform.runLater(() -> handleTwinLoaded(detail)),
                        error -> Platform.runLater(() -> handleLoadError(error)));
    }

    private void handleTwinLoaded(TwinDetailDTO detail) {
        try {
            // A. Cargar configuración en los controles
            loadConfigToUI(detail.config());

            // B. Hidratar la geometría localmente (optimización de red)
            this.currentGeometry = RiverGeometryFactory.createRealisticRiver(detail.config());

            // C. Actualizar visualización y motor físico
            drawRiverPreview();
            simEngine.loadRiver(currentGeometry, detail.config());

        } catch (Exception e) {
            log.error("Error procesando la geometría del río", e);
            showAlert(Alert.AlertType.ERROR, "Error de Visualización",
                    "Los datos son válidos, pero falló la generación gráfica.");
        }
    }

    private void handleLoadError(Throwable error) {
        log.error("Error recuperando detalles del Twin", error);
        showAlert(Alert.AlertType.ERROR, "Error de Conexión", "No se pudo cargar el río seleccionado.");
        onCancel();
    }

    private void updateManningLabel(Double n) {
        String text;
        if (n < 0.015)
            text = "Hormigón / Metal liso";
        else if (n < 0.025)
            text = "Tierra limpia / Arena";
        else if (n < 0.035)
            text = "Grava / Lecho natural estándar";
        else if (n < 0.050)
            text = "Rocas / Montaña / Algo de vegetación";
        else if (n < 0.075)
            text = "Vegetación densa / Troncos";
        else
            text = "Obstrucción extrema / Manglares";

        manningDescLabel.setText(text);

        if (n > 0.1)
            manningDescLabel.setStyle("-fx-text-fill: -color-danger-fg; -fx-font-size: 10px;");
        else
            manningDescLabel.setStyle("-fx-text-fill: -color-fg-muted; -fx-font-size: 10px;");
    }

    private void drawRiverPreview() {
        RiverConfig config = buildConfigFromUI();
        this.currentGeometry = RiverGeometryFactory.createRealisticRiver(config);
        drawHydrologyTab();
    }

    private void drawHydrologyTab() {
        this.currentGeometry = RiverGeometryFactory.createRealisticRiver(buildConfigFromUI());
        simEngine.loadRiver(currentGeometry, buildConfigFromUI());
        // Delegar pintado al Renderer
        renderer.render(currentGeometry, currentRenderMode, canvasInteractor.getMouseX(), canvasInteractor.getMouseY());
    }

    private void drawNoiseHeartBeat() {
        if (detailFreqSpinner.getValue() == null || zoneFreqSpinner.getValue() == null) {
            log.warn("Noise Spinners no inicializados. Saltando HeartBeat draw.");
            return;
        }
        NoiseSignatureRenderer.render(noiseCanvas, buildConfigFromUI());
    }

    // =========================================================================
    // 3. MAPEO DE DATOS (DTO <-> UI) - FASE 3: SUSTITUCIÓN DE LÓGICA
    // =========================================================================

    private void saveStateAsInitial(RiverConfig config, String name, String desc) {
        this.initialConfigState = config;
        this.initialNameState = name == null ? "" : name;
        this.initialDescState = desc == null ? "" : desc;
    }

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
        viewModel.loadFromConfig(config);

        eventPublisher.publishEvent(
                new TransitoryStatusUpdateEvent(
                        "Preconfiguración cargada!",
                        StatusViewModel.TransitionTime.IMMEDIATE,
                        StatusType.SUCCESS));
    }

    private RiverConfig buildConfigFromUI() {
        return viewModel.toDomainConfig();
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
        if (this.editingTwinId == null) {
            twinService.createTwin(request).subscribe(summary -> Platform.runLater(() -> {
                saveButton.setDisable(false);
                saveButton.setText("Crear Gemelo");
                showAlert(Alert.AlertType.INFORMATION, "Éxito",
                        "Gemelo Digital '" + summary.name() + "' creado correctamente.");

                eventPublisher.publishEvent(new TwinListRefreshEvent());
                eventPublisher.publishEvent(new SidebarVisibilityEvent(true));
                eventPublisher.publishEvent(new RestoreMainViewEvent());
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
        } else {
            twinService.updateTwin(editingTwinId, request).subscribe(summary -> Platform.runLater(() -> {
                saveButton.setDisable(false);
                saveButton.setText("Crear Gemelo");
                showAlert(Alert.AlertType.INFORMATION, "Éxito",
                        "Gemelo Digital '" + summary.name() + "' creado correctamente.");

                eventPublisher.publishEvent(new TwinListRefreshEvent());
                eventPublisher.publishEvent(new SidebarVisibilityEvent(true));
                eventPublisher.publishEvent(new RestoreMainViewEvent());
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
    }

    @FXML
    public void onCancel() {
        if (hasUnsavedChanges()) {
            Alert alert = new Alert(Alert.AlertType.CONFIRMATION);
            alert.setTitle("Cambios sin guardar");
            alert.setHeaderText("¿Estás seguro de que quieres salir?");
            alert.setContentText("Perderás la configuración actual del río.");

            if (alert.showAndWait().orElse(ButtonType.CANCEL) == ButtonType.CANCEL) {
                return;
            }
        }

        loadConfigToUI(RiverPresets.standard());
        nameField.clear();
        descField.clear();
        drawRiverPreview();

        eventPublisher.publishEvent(new TwinListRefreshEvent());
        eventPublisher.publishEvent(new SidebarVisibilityEvent(true));
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

    private void showAlert(Alert.AlertType type, String title, String content) {
        Alert alert = new Alert(type);
        alert.setTitle(title);
        alert.setHeaderText(null);
        alert.setContentText(content);
        alert.showAndWait();
    }
}