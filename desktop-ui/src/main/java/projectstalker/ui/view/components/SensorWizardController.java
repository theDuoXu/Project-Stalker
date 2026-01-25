package projectstalker.ui.view.components;

import javafx.application.Platform;
import javafx.collections.FXCollections;
import javafx.fxml.FXML;
import javafx.scene.control.*;
import javafx.scene.layout.StackPane;
import javafx.stage.Stage;
import javafx.util.StringConverter;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import projectstalker.domain.dto.sensor.SensorCreationDTO;
import projectstalker.domain.dto.twin.TwinSummaryDTO;
import projectstalker.domain.sensors.SensorType;
import projectstalker.ui.service.SensorClientService;
import projectstalker.ui.view.components.strategies.SensorUiStrategy;
import projectstalker.ui.view.components.strategies.SensorStrategyCategory;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Consumer;
import java.util.stream.Collectors;

@Slf4j
@Component
@RequiredArgsConstructor
public class SensorWizardController {

    private final SensorClientService sensorService;

    // --- INYECCIÓN DE ESTRATEGIAS ---
    // Spring busca todas las clases @Component que implementan la interfaz y las
    // mete en la lista
    private final List<SensorUiStrategy> availableStrategies;

    // --- State ---
    private TwinSummaryDTO twinContext;
    private double spatialResolutionKm;
    private Consumer<SensorCreationDTO> onSuccessCallback;

    // --- UI Elements ---
    @FXML
    private TextField nameField;
    @FXML
    private ComboBox<SensorType> sensorTypeCombo;
    @FXML
    private Slider locationSlider;
    @FXML
    private Label locationLabel;
    @FXML
    private Label cellLabel;

    @FXML
    private ToggleGroup strategyGroup;
    @FXML
    private ToggleButton manualToggle; // Injected
    @FXML
    private ToggleButton virtualToggle;
    @FXML
    private ToggleButton realToggle;

    // Selector Dinámico
    @FXML
    private ComboBox<SensorUiStrategy> strategySelector;
    @FXML
    private StackPane dynamicConfigContainer;

    @FXML
    private Button saveButton;

    @FXML
    private javafx.scene.chart.LineChart<Number, Number> previewChart;
    @FXML
    private javafx.scene.chart.NumberAxis xAxis;
    @FXML
    private javafx.scene.chart.NumberAxis yAxis;

    @FXML
    public void initialize() {
        setupCommonCombos();
        setupStrategyLogic();
        setupLocationLogic();
        startPreviewLoop();
    }

    private void startPreviewLoop() {
        // Loop de refresco de preview (cada 500ms) para detectar cambios en los
        // controles dinámicos
        javafx.animation.Timeline timeline = new javafx.animation.Timeline(
                new javafx.animation.KeyFrame(javafx.util.Duration.millis(500), ev -> updatePreview()));
        timeline.setCycleCount(javafx.animation.Animation.INDEFINITE);
        timeline.play();
    }

    private void updatePreview() {
        if (strategySelector.getValue() == null)
            return;
        SensorUiStrategy strategy = strategySelector.getValue();

        // Solo previsualizamos Estrategias Virtuales conocidas para no romper nada
        if (strategy.getCategory() != SensorStrategyCategory.VIRTUAL) {
            if (previewChart != null) // Avoid NPE if FXML failed inject
                previewChart.getData().clear();
            return;
        }

        if (previewChart == null)
            return;

        try {
            Map<String, Object> config = strategy.extractConfiguration();

            // Lógica Específica para SineWave (Hardcoded support for demo)
            if (config.containsKey("amplitude") && config.containsKey("frequency")) {
                double amp = Double.parseDouble(config.getOrDefault("amplitude", "0").toString());
                double freq = Double.parseDouble(config.getOrDefault("frequency", "0").toString());
                double offset = Double.parseDouble(config.getOrDefault("offset", "0").toString());

                javafx.scene.chart.XYChart.Series<Number, Number> series = new javafx.scene.chart.XYChart.Series<>();
                series.setName("Simulación");

                // Generar 60 segundos
                for (double t = 0; t <= 60; t += 0.5) {
                    double val = offset + amp * Math.sin(freq * t);
                    series.getData().add(new javafx.scene.chart.XYChart.Data<>(t, val));
                }

                previewChart.getData().clear();
                previewChart.getData().add(series);
            }
        } catch (Exception e) {
            // Ignore preview errors during typing
        }
    }

    public void setTwinContext(TwinSummaryDTO twin) {
        Objects.requireNonNull(twin);
        this.twinContext = twin;
        double lengthKm = twin.totalLengthKm();
        double dxMeters = twin.totalLengthKm() / twin.cellCount();

        if (lengthKm <= 0 || dxMeters <= 0)
            throw new IllegalArgumentException(
                    "La longitud o la resolución espacial no pueden ser menor o igual a cero");

        this.spatialResolutionKm = dxMeters / 1000.0;
        locationSlider.setMax(lengthKm);
        locationSlider.setMajorTickUnit(Math.max(5.0, lengthKm / 10.0));
    }

    public void setOnSuccess(Consumer<SensorCreationDTO> cb) {
        this.onSuccessCallback = cb;
    }

    private void setupLocationLogic() {
        locationSlider.valueProperty().addListener((obs, old, val) -> {
            if (spatialResolutionKm <= 0)
                return;
            int cellIndex = (int) (val.doubleValue() / spatialResolutionKm);
            locationLabel.setText(String.format("Km %.2f", val.doubleValue()));
            cellLabel.setText("Celda: " + cellIndex);
        });
    }

    private void setupStrategyLogic() {
        // Renderizador del ComboBox de Estrategias
        strategySelector.setConverter(new StringConverter<>() {
            @Override
            public String toString(SensorUiStrategy s) {
                return s == null ? "" : s.getDisplayName();
            }

            @Override
            public SensorUiStrategy fromString(String id) {
                return null;
            }
        });

        // Al cambiar Virtual/Real, filtramos la lista
        strategyGroup.selectedToggleProperty().addListener((obs, old, val) -> filterStrategies());

        // Al seleccionar una estrategia, renderizamos su panel
        strategySelector.valueProperty().addListener((obs, oldS, newS) -> {
            dynamicConfigContainer.getChildren().clear();
            if (newS != null) {
                newS.reset();
                dynamicConfigContainer.getChildren().add(newS.render());
            }
        });

        // Estado inicial
        filterStrategies();
    }

    // ...

    private void filterStrategies() {
        if (strategyGroup.getSelectedToggle() == null)
            return;

        SensorStrategyCategory targetCategory;
        if (manualToggle.isSelected())
            targetCategory = SensorStrategyCategory.MANUAL;
        else if (realToggle.isSelected())
            targetCategory = SensorStrategyCategory.REAL;
        else
            targetCategory = SensorStrategyCategory.VIRTUAL;

        List<SensorUiStrategy> filtered = availableStrategies.stream()
                .filter(s -> s.getCategory() == targetCategory)
                .collect(Collectors.toList());

        strategySelector.setItems(FXCollections.observableArrayList(filtered));
        if (!filtered.isEmpty())
            strategySelector.getSelectionModel().selectFirst();
    }

    // --- Edit Mode ---
    private String editingSensorId = null;

    @FXML
    public void onSave() {
        SensorUiStrategy currentStrategy = strategySelector.getValue();

        // 1. Validaciones
        if (nameField.getText().isBlank()) {
            showAlert("Nombre requerido");
            return;
        }
        if (sensorTypeCombo.getValue() == null) {
            showAlert("Tipo requerido");
            return;
        }
        if (currentStrategy == null) {
            showAlert("Estrategia requerida");
            return;
        }

        // Delegamos la validación específica a la estrategia
        if (!currentStrategy.validate()) {
            showAlert("Error en parámetros de la estrategia. Revise los campos numéricos o requeridos.");
            return;
        }

        try {
            // 2. Construcción DTO Genérica
            SensorCreationDTO dto = buildDto(currentStrategy);

            saveButton.setDisable(true);

            if (editingSensorId == null) {
                // CREATE
                sensorService.createSensor(dto)
                        .subscribe(r -> Platform.runLater(() -> {
                            if (onSuccessCallback != null)
                                onSuccessCallback.accept(dto);
                            closeWindow();
                        }), e -> Platform.runLater(() -> {
                            saveButton.setDisable(false);
                            showAlert("Error Backend: " + e.getMessage());
                        }));
            } else {
                // UPDATE
                sensorService.updateSensor(editingSensorId, dto)
                        .subscribe(r -> Platform.runLater(() -> {
                            if (onSuccessCallback != null)
                                onSuccessCallback.accept(dto);
                            closeWindow();
                        }), e -> Platform.runLater(() -> {
                            saveButton.setDisable(false);
                            showAlert("Error Backend: " + e.getMessage());
                        }));
            }

        } catch (Exception e) {
            log.error("Error", e);
        }
    }

    /**
     * Pre-fills the wizard with existing sensor data for editing.
     */
    public void setSensorToEdit(projectstalker.domain.dto.sensor.SensorResponseDTO existing) {
        this.editingSensorId = existing.stationId();
        this.nameField.setText(existing.name());
        this.saveButton.setText("Actualizar");

        // 1. Set Type
        // Attempt to fuzzy match type from code/unit if not strictly available,
        // but now we have typeCode in DTO!
        String tCode = existing.typeCode();
        if (tCode != null) {
            for (SensorType t : SensorType.getValidTypes()) {
                if (t.getCode().equalsIgnoreCase(tCode)) {
                    sensorTypeCombo.getSelectionModel().select(t);
                    break;
                }
            }
        }

        // 2. Select Strategy
        // signalType in DTO is now strategyType (e.g. VIRTUAL_PHYSICS_FLOW)
        String strategyCode = existing.signalType();
        if (strategyCode != null) {
            for (SensorUiStrategy s : availableStrategies) { // Changed from 'strategies' to 'availableStrategies'
                if (s.getStrategyCode().equalsIgnoreCase(strategyCode)) {
                    // First, ensure the correct category is selected to make the strategy visible
                    // This assumes strategyGroup and filterStrategies are working correctly
                    if (s.getCategory() == SensorStrategyCategory.MANUAL) {
                        manualToggle.setSelected(true);
                    } else if (s.getCategory() == SensorStrategyCategory.REAL) {
                        realToggle.setSelected(true);
                    } else {
                        virtualToggle.setSelected(true);
                    }
                    filterStrategies(); // Re-filter to ensure the strategy is in the combo box
                    strategySelector.getSelectionModel().select(s);
                    // 3. Populate Config
                    s.populate(existing.configuration());
                    break;
                }
            }
        }
    }

    private SensorCreationDTO buildDto(SensorUiStrategy strategy) {
        // Extraemos configuración de la estrategia
        Map<String, Object> config = strategy.extractConfiguration();

        // Añadimos metadatos del contexto (Legacy/Redundant but harmless to keep in map
        // for now)
        config.put("twinId", twinContext.id());
        config.put("strategy", strategy.getStrategyCode());

        // Mapping Category to StrategyType string expected by Backend
        // "VIRTUAL" or "REAL" or "MANUAL" (New)
        String strategyTypeStr = strategy.getCategory().name();

        return new SensorCreationDTO(
                nameField.getText(),
                sensorTypeCombo.getValue().getCode(),
                locationSlider.getValue(),
                strategyTypeStr,
                twinContext.id(), // New explicit field
                config);
    }

    private void setupCommonCombos() {
        sensorTypeCombo.setItems(FXCollections.observableArrayList(SensorType.getValidTypes()));
        sensorTypeCombo.setConverter(new StringConverter<>() {
            @Override
            public String toString(SensorType object) {
                return object == null ? "" : object.getFriendlyName();
            }

            @Override
            public SensorType fromString(String string) {
                return null;
            }
        });
    }

    @FXML
    public void onCancel() {
        closeWindow();
    }

    private void closeWindow() {
        ((Stage) nameField.getScene().getWindow()).close();
    }

    private void showAlert(String m) {
        new Alert(Alert.AlertType.WARNING, m).showAndWait();
    }
}