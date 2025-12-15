package projectstalker.ui.view;

import javafx.application.Platform;
import javafx.beans.value.ObservableValue;
import javafx.collections.FXCollections;
import javafx.fxml.FXML;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.XYChart;
import javafx.scene.control.*;
import javafx.scene.paint.Color;
import javafx.util.StringConverter;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.ApplicationEventPublisher;
import org.springframework.stereotype.Component;
import projectstalker.config.RiverConfig;
import projectstalker.domain.dto.twin.FlowPreviewRequest;
import projectstalker.domain.dto.twin.TwinCreateRequest;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.factory.RiverGeometryFactory;
import projectstalker.ui.event.RestoreMainViewEvent;
import projectstalker.ui.event.SidebarVisibilityEvent;
import projectstalker.ui.renderer.RiverRenderer;
import projectstalker.ui.service.DigitalTwinClientService;
import projectstalker.ui.view.util.RiverPresets;
import projectstalker.utils.FastNoiseLite;

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
    private double lastMouseX = -1;
    private double lastMouseY = -1;

    // Para control de cambios
    private RiverConfig initialConfigState;
    private String initialNameState = "";
    private String initialDescState = "";

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
    public Canvas noiseCanvas;
    @FXML
    public LineChart<Number, Number> previewChart;
    @FXML
    public Tab morphologyTab;
    @FXML
    public Tab noiseTab;
    @FXML
    public Button saveButton;
    @FXML
    public ToggleButton morphologySwitch;

    public RiverEditorController(DigitalTwinClientService twinService, ApplicationEventPublisher eventPublisher) {
        this.twinService = twinService;
        this.eventPublisher = eventPublisher;
    }

    @FXML
    public void initialize() {
        this.renderer = new RiverRenderer(morphologyCanvas);
        setupRendererListeners();
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

        setupGeometryControls();
        setupNoiseControls();
        setupCanvasResizing();
    }

    private void setupMorphologySwitch() {
        morphologySwitch.selectedProperty().addListener((observable, oldValue, newValue) -> {
            onMorphologyCanvasModeChange(newValue);
        });
    }

    private void setupRendererListeners() {
        morphologyCanvas.setOnMouseMoved(e -> {
            this.lastMouseX = e.getX();
            this.lastMouseY = e.getY();
            if (currentGeometry != null) {
                renderer.render(currentGeometry, currentRenderMode, lastMouseX, lastMouseY);
            }
        });

        morphologyCanvas.setOnMouseExited(e -> {
            this.lastMouseX = -1;
            this.lastMouseY = -1;
            // Pasamos -1 para ocultar el HUD
            if (currentGeometry != null) renderer.render(currentGeometry, currentRenderMode, lastMouseX, lastMouseY);
        });

        morphologyCanvas.sceneProperty().addListener((obs, oldVal, newVal) -> {
            if (newVal != null) {
                // Pequeño delay para asegurar que el CSS se ha procesado
                javafx.application.Platform.runLater(() -> {
                    renderer.reloadThemeColors();
                    if (currentGeometry != null)
                        renderer.render(currentGeometry, currentRenderMode, lastMouseX, lastMouseY);
                });
            }
        });
    }


    // =========================================================================
    // 1. CONFIGURACIÓN DE CONTROLES (SETUP)
    // =========================================================================

    private void setupCanvasResizing() {
        // El canvas debe redibujarse si cambia el tamaño del contenedor padre
        noiseCanvas.widthProperty().addListener(evt -> drawNoiseHeartBeat());
        noiseCanvas.heightProperty().addListener(evt -> drawNoiseHeartBeat());
    }

    private void setupGeometrySpinners() {
        // Longitud: 1km a 500km, paso de 10m
        totalLengthSpinner.setValueFactory(new SpinnerValueFactory.DoubleSpinnerValueFactory(1000, 500000, 50000, 10));

        // Ancho: 5m a 2km, paso de 5m
        baseWidthSpinner.setValueFactory(new SpinnerValueFactory.DoubleSpinnerValueFactory(5, 2000, 150, 5));
        varWidthSpinner.setValueFactory(new SpinnerValueFactory.DoubleSpinnerValueFactory(0, 100, 10, 0.1));

        // Pendiente: 0.00001 a 0.1, paso fino de 0.0001
        var slopeFactory = new SpinnerValueFactory.DoubleSpinnerValueFactory(0.00001, 0.1, 0.0002, 0.0001);
        slopeFactory.setConverter(createConverter("%.5f", 0.0002));
        slopeSpinner.setValueFactory(slopeFactory);
        varSlopeSpinner.setValueFactory(new SpinnerValueFactory.DoubleSpinnerValueFactory(0, 100, 10, 0.1));
    }

    private void setupManningSpinner() {
        // Rango científico: 0.010 (Canal liso) a 0.150 (Máximo razonable). Paso: 0.001
        var factory = new SpinnerValueFactory.DoubleSpinnerValueFactory(0.010, 0.150, 0.030, 0.001);
        factory.setConverter(createConverter("%.3f", 0.030));

        manningSpinner.setValueFactory(factory);
        varManningSpinner.setValueFactory(new SpinnerValueFactory.DoubleSpinnerValueFactory(0, 100, 10, 0.1));

        // Listener para dar feedback textual sobre el tipo de material
        manningSpinner.valueProperty().addListener((obs, oldVal, newVal) -> updateManningLabel(newVal));
    }

    private void setupNoiseSpinners() {
        // 1. Seed (Semilla) - Entero Grande
        // Rango: 0 a 999999999. Paso: 1 o 10000. Usamos Long para la seguridad del tipo.
        seedSpinner.setValueFactory(new SpinnerValueFactory.IntegerSpinnerValueFactory(0, 9999999, 1000, 1));

        // 2. Frecuencia de Detalle (detailFreqSpinner) - Valor fino
        // Rango: 0.001 a 1.0. Paso: 0.001 (muy fino).
        var detailFactory = new SpinnerValueFactory.DoubleSpinnerValueFactory(0.001, 1.0, 0.05, 0.001);
        detailFactory.setConverter(createConverter("%.4f", 0.05));
        detailFreqSpinner.setValueFactory(detailFactory);

        // 3. Frecuencia Zonal (zoneFreqSpinner) - Valor muy fino (escala macro)
        // Rango: 0.0001 a 0.1. Paso: 0.0001 (aún más fino).
        var zoneFactory = new SpinnerValueFactory.DoubleSpinnerValueFactory(0.0001, 0.1, 0.001, 0.0001);
        zoneFactory.setConverter(createConverter("%.5f", 0.001));
        zoneFreqSpinner.setValueFactory(zoneFactory);
    }
    private void setupAdvancedSpinners() {
        // Concavidad (0.4 default)
        concavitySpinner.setValueFactory(new SpinnerValueFactory.DoubleSpinnerValueFactory(0.0, 1.0, 0.4, 0.05));

        // Talud (z) (4.0 default)
        sideSlopeSpinner.setValueFactory(new SpinnerValueFactory.DoubleSpinnerValueFactory(0.1, 10.0, 4.0, 0.5));

        // Sensibilidad Ancho-Pendiente (0.4 default)
        slopeSensSpinner.setValueFactory(new SpinnerValueFactory.DoubleSpinnerValueFactory(0.1, 1.0, 0.4, 0.05));

        // Tasa Decay k20 (0.1 default)
        decayRateSpinner.setValueFactory(new SpinnerValueFactory.DoubleSpinnerValueFactory(0.0, 2.0, 0.1, 0.01));

        // Sensibilidad Turbulencia (0.8 default)
        turbSensSpinner.setValueFactory(new SpinnerValueFactory.DoubleSpinnerValueFactory(0.0, 2.0, 0.8, 0.1));

        // Enfriamiento Cabecera (4.0 default)
        headwaterCoolingSpinner.setValueFactory(new SpinnerValueFactory.DoubleSpinnerValueFactory(0.0, 15.0, 4.0, 0.5));

        // Factor Calentamiento por Ancho (1.5 default)
        widthHeatingSpinner.setValueFactory(new SpinnerValueFactory.DoubleSpinnerValueFactory(0.0, 5.0, 1.5, 0.1));

        // Listeners para actualizar UI si quieres reactividad inmediata
        concavitySpinner.valueProperty().addListener(this::onUpdateGeometryOrNoiseSpinners);
        sideSlopeSpinner.valueProperty().addListener(this::onUpdateGeometryOrNoiseSpinners);
        slopeSensSpinner.valueProperty().addListener(this::onUpdateGeometryOrNoiseSpinners);
        decayRateSpinner.valueProperty().addListener(this::onUpdateGeometryOrNoiseSpinners);

        // TODO placeholders
        headwaterCoolingSpinner.valueProperty().addListener(this::onUpdateGeometryOrNoiseSpinners);
        widthHeatingSpinner.valueProperty().addListener(this::onUpdateGeometryOrNoiseSpinners);

    }

    private void setupGeometryControls() {
        // --- GEOMETRÍA (Base y Variabilidad) ---
        totalLengthSpinner.valueProperty().addListener(this::onUpdateGeometryOrNoiseSpinners);
        baseWidthSpinner.valueProperty().addListener(this::onUpdateGeometryOrNoiseSpinners);
        varWidthSpinner.valueProperty().addListener(this::onUpdateGeometryOrNoiseSpinners);
        slopeSpinner.valueProperty().addListener(this::onUpdateGeometryOrNoiseSpinners);
        varSlopeSpinner.valueProperty().addListener(this::onUpdateGeometryOrNoiseSpinners);

        // --- HIDRÁULICA (Base y Variabilidad) ---
        manningSpinner.valueProperty().addListener(this::onUpdateGeometryOrNoiseSpinners);
        varManningSpinner.valueProperty().addListener(this::onUpdateGeometryOrNoiseSpinners);


        morphologyTab.setOnSelectionChanged(event -> {
            if (morphologyTab.isSelected()) {
                highlightTab(morphologyTab, false);
                drawRiverPreview();
            }
        });
        // Dibujo inicial
        if (morphologyTab.isSelected()) {
            drawRiverPreview();
        }
    }


    private void setupNoiseControls() {
        // El Slider controla la frecuencia principal visualmente (0-100)
        // Lo mapeamos a una frecuencia pequeña (0.0 - 0.05)
        noiseSlider.valueProperty().addListener((obs, oldVal, newVal) -> {
            double freq = newVal.doubleValue() / 100;
            noiseFreqValueLabel.setText(String.format("%.5f", freq));
            // Si estamos en la pestaña de morfología, redibujamos en tiempo real
            drawNoiseHeartBeat();
            if (!noiseTab.isSelected()) {
                highlightTab(noiseTab, true);
                drawRiverPreview();
            }
        });

        seedSpinner.valueProperty().addListener(this::onUpdateGeometryOrNoiseSpinners);
        seedSpinner.valueProperty().addListener(this::onUpdateNoiseSpinners);
        // Listeners para redibujar si cambian los campos manuales de texto
        detailFreqSpinner.valueProperty().addListener(this::onUpdateGeometryOrNoiseSpinners);
        detailFreqSpinner.valueProperty().addListener(this::onUpdateNoiseSpinners);

        zoneFreqSpinner.valueProperty().addListener(this::onUpdateGeometryOrNoiseSpinners);
        zoneFreqSpinner.valueProperty().addListener(this::onUpdateNoiseSpinners);

        // Listener para LIMPIAR el resaltado cuando el usuario selecciona la pestaña.
        noiseTab.setOnSelectionChanged(event -> {
            if (noiseTab.isSelected()) {
                highlightTab(noiseTab, false);
                drawNoiseHeartBeat();
            }
        });
        // Dibujo inicial
        if (noiseTab.isSelected()) {
            drawNoiseHeartBeat();
        }
    }

    private void onUpdateGeometryOrNoiseSpinners(ObservableValue<? extends Number> o, Number old, Number val) {
        // Lógica de enrutamiento inteligente: Redibujar lo que el usuario está viendo
        if (morphologyTab.isSelected()) {
            drawRiverPreview();
        } else if (noiseTab.isSelected()) {
            drawNoiseHeartBeat();
        }

        // Gestión de notificaciones (Asteriscos)
        if (!morphologyTab.isSelected()) highlightTab(morphologyTab, true);
    }

    private void onUpdateNoiseSpinners(ObservableValue<? extends Number> o, Number old, Number val) {
        if (!noiseTab.isSelected()) highlightTab(noiseTab, true);
    }

    private void highlightTab(Tab tab, boolean highlight) {
        if (highlight) {
            // Añade un símbolo Unicode y un estilo para llamar la atención.
            if (!tab.getText().contains("✱")) {
                tab.setText(tab.getText() + " ✱");
            }
            tab.setStyle("-fx-background-color: #FFA50040; -fx-text-fill: -color-accent-fg;"); // Fondo naranja semi-transparente
        } else {
            // Limpiar el estilo y el símbolo.
            tab.setStyle(null);
            tab.setText(tab.getText().replace(" ✱", ""));
        }
    }

    private void setupPhysicoChemical() {
        dailyBaseTempSpinner.setValueFactory(new SpinnerValueFactory.DoubleSpinnerValueFactory(-5.0, 40.0, 15.0, 0.1));
        varDailyBaseTempSpinner.setValueFactory(new SpinnerValueFactory.DoubleSpinnerValueFactory(0, 100, 20.0, 0.1));

        anualBaseTempSpinner.setValueFactory(new SpinnerValueFactory.DoubleSpinnerValueFactory(-5.0, 40.0, 15.0, 0.1));
        varAnualBaseTempSpinner.setValueFactory(new SpinnerValueFactory.DoubleSpinnerValueFactory(0, 100, 10.0, 0.1));

        basePhSpinner.setValueFactory(new SpinnerValueFactory.DoubleSpinnerValueFactory(0.0, 14.0, 7.5, 0.1));
        varBasePhSpinner.setValueFactory(new SpinnerValueFactory.DoubleSpinnerValueFactory(0, 50, 10.0, 0.1));

        dispersionSpinner.setValueFactory(new SpinnerValueFactory.DoubleSpinnerValueFactory(0.1, 100.0, 10.0, 1.0));
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

        // Delegar pintado al Renderer
        renderer.render(currentGeometry, currentRenderMode, lastMouseX, lastMouseY);
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

        GraphicsContext gc = noiseCanvas.getGraphicsContext2D();
        double w = noiseCanvas.getWidth();
        double h = noiseCanvas.getHeight();

        // Limpiar fondo
        gc.setFill(Color.web("#2E3440"));
        gc.fillRect(0, 0, w, h);

        // 1. Obtención de Parámetros
        long seed = seedSpinner.getValue();
        double mainFreq = noiseSlider.getValue() / 2000;
        double detailFreq = detailFreqSpinner.getValue();
        double zoneFreq = zoneFreqSpinner.getValue();

        // 2. Inicializar los Generadores de Ruido

        // Generador PRINCIPAL (usa la frecuencia del Slider)
        final FastNoiseLite mainNoise = new FastNoiseLite((int) seed); // Usa la semilla base
        mainNoise.SetNoiseType(FastNoiseLite.NoiseType.Perlin);
        mainNoise.SetFrequency((float) mainFreq);

        // Generador de Detalle
        final FastNoiseLite detailNoise = new FastNoiseLite((int) seed + 1);
        detailNoise.SetNoiseType(FastNoiseLite.NoiseType.Perlin);
        detailNoise.SetFrequency((float) detailFreq);

        // Generador de Zona
        final FastNoiseLite zoneNoise = new FastNoiseLite((int) seed + 2);
        zoneNoise.SetNoiseType(FastNoiseLite.NoiseType.Perlin);
        zoneNoise.SetFrequency((float) zoneFreq);

        // 3. Configuración de Dibujo
        gc.setStroke(Color.web("#88C0D0"));
        gc.setLineWidth(2.0);
        gc.beginPath();

        double yBase = h / 2;
        double maxAmplitude = h * 0.45;
        double scaleX = 0.5;

        for (int x = 0; x < w; x++) {
            double worldX = x * scaleX;
            int i = (int) worldX;

            // Obtener los valores de ruido normalizados a [-1, 1]
            double currentMainNoise = mainNoise.GetNoise(i, 0);
            double currentDetailNoise = detailNoise.GetNoise(i, 0);
            double currentZoneNoise = zoneNoise.GetNoise(i, 0);

            // Generar la forma de onda combinada (Ponderación)
            // Damos más peso al ruido principal y zonal, y menos al detalle.
            // La suma debe ser aproximada a 1.0 para mantener la amplitud.
            double noiseEffect = (currentMainNoise * 0.6) +  // Principal (Slider)
                    (currentZoneNoise * 0.3) +  // Zonal (Frecuencia Macro)
                    (currentDetailNoise * 0.1); // Detalle (Frecuencia Fina)

            // Normalizar (ya que noiseEffect ahora está escalado)
            double yFinal = yBase + (noiseEffect * maxAmplitude);

            if (x == 0) gc.moveTo(x, yFinal);
            else gc.lineTo(x, yFinal);
        }
        gc.stroke();

        // Overlay de texto
        gc.setFill(Color.WHITE);
        gc.fillText("Firma de Ruido (Seed: " + seed + ")", 10, 20);
    }

    // Helper para parsear el Long del seedField (Necesario en el controlador)
    private long parseSafeLong(String txt, long def) {
        try {
            return Long.parseLong(txt);
        } catch (Exception e) {
            return def;
        }
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
                .withSeed(seedSpinner.getValue())
                .withTotalLength(totalLengthSpinner.getValue().floatValue())
                .withBaseWidth(baseWidthSpinner.getValue().floatValue())
                .withWidthVariability(
                        (float) (varWidthSpinner.getValue() / 100 * baseWidthSpinner.getValue())
                ) // Cálculo inverso: varWidth = ratio * baseWidth
                .withAverageSlope(slopeSpinner.getValue().floatValue())
                .withSlopeVariability(
                        (float) (varSlopeSpinner.getValue() / 100 * slopeSpinner.getValue())
                ) // Cálculo inverso: varSlope = ratio * avgSlope

                // --- 2. HIDRÁULICA ---
                .withBaseManning(manningSpinner.getValue().floatValue())
                .withManningVariability(
                        (float) (varManningSpinner.getValue() / 100 * manningSpinner.getValue())
                ) // Cálculo inverso: varManning = ratio * baseManning

                // --- 3. PROCEDURAL (Ruido) ---
                .withNoiseFrequency(mainFreq)
                .withDetailNoiseFrequency(detailFreqSpinner.getValue().floatValue())
                .withZoneNoiseFrequency(zoneFreqSpinner.getValue().floatValue())

                // --- 4. FÍSICO-QUÍMICA (Temperatura, pH, Dispersión) ---
                // Temperatura
                .withDailyBaseTemperature(dailyBaseTempSpinner.getValue().floatValue())
                .withDailyTempVariation(
                        (float) (varDailyBaseTempSpinner.getValue() / 100 * dailyBaseTempSpinner.getValue())
                )
                .withAverageAnualTemperature(anualBaseTempSpinner.getValue().floatValue())
                .withSeasonalTempVariation(
                        (float) (varAnualBaseTempSpinner.getValue() / 100 * anualBaseTempSpinner.getValue())
                )

                // pH
                .withBasePh(basePhSpinner.getValue().floatValue())
                .withPhVariability(
                        (float) (varBasePhSpinner.getValue() / 100 * basePhSpinner.getValue())
                ) // Cálculo inverso: varPh = ratio * basePh

                // Dispersión
                .withBaseDispersionAlpha(dispersionSpinner.getValue().floatValue())

                // Avanzados
                .withConcavityFactor(concavitySpinner.getValue().floatValue())
                .withBaseSideSlope(sideSlopeSpinner.getValue().floatValue())
                .withSlopeSensitivityExponent(slopeSensSpinner.getValue().floatValue())
                .withBaseDecayRateAt20C(decayRateSpinner.getValue().floatValue())
                .withDecayTurbulenceSensitivity(turbSensSpinner.getValue().floatValue())
                .withMaxHeadwaterCoolingEffect(headwaterCoolingSpinner.getValue().floatValue())
                .withWidthHeatingFactor(widthHeatingSpinner.getValue().floatValue());
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
    public void onPreviewClick() {
        // 1. Asegurar que estamos viendo la pestaña correcta
        previewTabs.getSelectionModel().select(1);

        // 2. Construir la request
        RiverConfig config = buildConfigFromUI();
        var request = new FlowPreviewRequest(config.seed(), 100.0f, // Caudal base dummy
                0.2f,   // Variabilidad dummy
                0.1f, 300    // Segundos a simular
        );

        // 3. Limpiar chart previo
        previewChart.getData().clear();

        // 4. Llamada reactiva
        twinService.previewFlow(request).subscribe(data -> {
            Platform.runLater(() -> updateChart(data));
        }, error -> {
            Platform.runLater(() -> showAlert(Alert.AlertType.ERROR, "Error Simulación", error.getMessage()));
        });
    }

    @FXML
    private void onMorphologyCanvasModeChange(Boolean newValue) {
        currentRenderMode = RiverRenderer.RenderMode.fromBoolean(newValue);
    }

    private void updateChart(float[] flowData) {
        XYChart.Series<Number, Number> series = new XYChart.Series<>();
        series.setName("Hidrograma Generado");

        // Decimación simple para rendimiento de UI
        int step = Math.max(1, flowData.length / 300);

        for (int i = 0; i < flowData.length; i += step) {
            series.getData().add(new XYChart.Data<>(i, flowData[i]));
        }

        previewChart.getData().add(series);
    }

    // =========================================================================
    // 5. HELPERS
    // =========================================================================

    /**
     * Helper limpio para crear convertidores de String a Double.
     * Evita el uso de clases anónimas sucias y maneja NumberFormatException.
     */
    private StringConverter<Double> createConverter(String format, double fallbackValue) {
        return new StringConverter<>() {
            @Override
            public String toString(Double object) {
                if (object == null) return String.format(format, fallbackValue);
                return String.format(format, object);
            }

            @Override
            public Double fromString(String string) {
                try {
                    return Double.parseDouble(string.replace(",", ".")); // Tolerancia a comas
                } catch (NumberFormatException e) {
                    return fallbackValue;
                }
            }
        };
    }

    private double parseSafeDouble(String txt, double def) {
        try {
            return Double.parseDouble(txt);
        } catch (Exception e) {
            return def;
        }
    }

    private void showAlert(Alert.AlertType type, String title, String content) {
        Alert alert = new Alert(type);
        alert.setTitle(title);
        alert.setHeaderText(null);
        alert.setContentText(content);
        alert.showAndWait();
    }


}