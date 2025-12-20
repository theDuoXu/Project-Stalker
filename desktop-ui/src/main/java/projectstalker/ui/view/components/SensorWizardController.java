package projectstalker.ui.view.components;

import javafx.application.Platform;
import javafx.beans.binding.Bindings;
import javafx.collections.FXCollections;
import javafx.fxml.FXML;
import javafx.scene.control.*;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;
import javafx.util.StringConverter;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import projectstalker.domain.dto.sensor.SensorCreationDTO;
import projectstalker.domain.dto.twin.TwinSummaryDTO; // <--- USAMOS EL DTO
import projectstalker.domain.sensors.SensorType;
import projectstalker.ui.service.SensorClientService;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.function.Consumer;

@Slf4j
@Component
@RequiredArgsConstructor
public class SensorWizardController {

    private final SensorClientService sensorService;

    // --- State ---
    private TwinSummaryDTO twinContext; // <--- Referencia al objeto de dominio
    private double spatialResolutionKm; // Cacheado para cálculos
    private Consumer<SensorCreationDTO> onSuccessCallback;

    // --- UI Elements ---
    @FXML private TextField nameField;
    @FXML private ComboBox<SensorType> sensorTypeCombo;
    @FXML private Slider locationSlider;
    @FXML private Label locationLabel;
    @FXML private Label cellLabel;
    @FXML private Button saveButton;

    // ... (Resto de injects de ToggleGroups y Paneles igual que antes) ...
    @FXML private ToggleGroup strategyGroup;
    @FXML private ToggleButton virtualToggle;
    @FXML private VBox virtualPanel;
    @FXML private VBox realPanel;
    @FXML private ComboBox<String> virtualStrategyCombo;
    @FXML private GridPane paramsRandomPanel;
    @FXML private GridPane paramsSinePanel;
    @FXML private TextField baseValueField;
    @FXML private Slider volatilitySlider;
    @FXML private TextField sineOffsetField;
    @FXML private TextField sineAmplitudeField;
    @FXML private TextField sineFreqField;
    @FXML private TextField webhookUrlField;
    @FXML private PasswordField apiTokenField;

    @FXML
    public void initialize() {
        setupCombos();
        setupStrategySwitch();
        setupLocationLogic();
    }

    /**
     * Inyecta el contexto del Gemelo Digital.
     * FAIL FAST: Si el DTO es inválido, explota aquí, no después.
     */
    public void setTwinContext(TwinSummaryDTO twin) {
        // 1. Fail Fast Validation
        Objects.requireNonNull(twin, "El contexto del Twin no puede ser NULL");
        if (twin.id() == null || twin.id().isBlank()) {
            throw new IllegalArgumentException("El Twin DTO tiene un ID inválido");
        }

        // 2. Asignación
        this.twinContext = twin;

        // 3. Configuración Física (Defaults defensivos si vienen a 0)
        double lengthKm = twin.totalLengthKm() > 0 ? twin.totalLengthKm() : 10.0;
        double dxMeters = twin.spatialResolutionMeters() > 0 ? twin.spatialResolutionMeters() : 100.0;

        this.spatialResolutionKm = dxMeters / 1000.0;

        // 4. Ajuste de UI
        locationSlider.setMax(lengthKm);
        double tickUnit = Math.max(5.0, lengthKm / 10.0);
        locationSlider.setMajorTickUnit(tickUnit);
        locationSlider.setBlockIncrement(this.spatialResolutionKm);

        log.debug("Wizard inicializado para Twin: '{}' (Len: {}km, dx: {}m)", twin.name(), lengthKm, dxMeters);
    }

    public void setOnSuccess(Consumer<SensorCreationDTO> callback) {
        this.onSuccessCallback = callback;
    }

    // ... (setupCombos y setupStrategySwitch se mantienen igual) ...

    private void setupLocationLogic() {
        locationSlider.valueProperty().addListener((obs, oldVal, kmVal) -> {
            // Protección contra división por cero si spatialResolutionKm no se ha seteado aún
            if (spatialResolutionKm <= 0) return;

            double km = kmVal.doubleValue();

            // Cálculo de celda usando el contexto físico real
            int cellIndex = (int) (km / spatialResolutionKm);
            int maxCells = (int) (locationSlider.getMax() / spatialResolutionKm);

            if (cellIndex >= maxCells) cellIndex = maxCells - 1;

            locationLabel.setText(String.format("Km %.3f", km));
            cellLabel.setText(String.format("Celda ID: #%d", cellIndex));
        });
    }

    @FXML
    public void onSave() {
        if (!validateForm()) return;

        // Doble check de seguridad (Fail Fast)
        if (twinContext == null) {
            throw new IllegalStateException("Intentando guardar sin contexto de Twin inicializado.");
        }

        try {
            SensorCreationDTO dto = buildSensorDTO();
            saveButton.setDisable(true);

            sensorService.createSensor(dto)
                    .subscribe(response -> Platform.runLater(() -> {
                        log.info("Sensor '{}' creado en celda {}", response.name(), dto.configuration().get("twinId")); // solo log
                        if (onSuccessCallback != null) onSuccessCallback.accept(dto);
                        closeWindow();
                    }), error -> Platform.runLater(() -> {
                        saveButton.setDisable(false);
                        showAlert("Error Backend", error.getMessage());
                    }));

        } catch (Exception e) {
            log.error("Error local", e);
            showAlert("Error Local", e.getMessage());
        }
    }

    private SensorCreationDTO buildSensorDTO() {
        boolean isVirtual = virtualToggle.isSelected();
        Map<String, Object> config = new HashMap<>();

        // Usamos el DTO almacenado
        config.put("twinId", twinContext.id());

        if (isVirtual) {
            // ... (Lógica de mapeo virtual igual que antes) ...
            String subStrategy = virtualStrategyCombo.getValue();
            if (subStrategy != null && subStrategy.contains("Seno")) {
                config.put("strategy", "VIRTUAL_SINE");
                config.put("offset", Double.parseDouble(sineOffsetField.getText()));
                config.put("amplitude", Double.parseDouble(sineAmplitudeField.getText()));
                config.put("frequency", Double.parseDouble(sineFreqField.getText()));
            } else {
                config.put("strategy", "VIRTUAL_RANDOM_WALK");
                config.put("baseValue", Double.parseDouble(baseValueField.getText()));
                config.put("volatility", volatilitySlider.getValue());
            }
        } else {
            config.put("strategy", "REAL_IoT_WEBHOOK");
            config.put("url", webhookUrlField.getText());
            config.put("token", apiTokenField.getText());
        }

        return new SensorCreationDTO(
                nameField.getText(),
                sensorTypeCombo.getValue().getCode(),
                locationSlider.getValue(),
                isVirtual ? "VIRTUAL" : "REAL",
                config
        );
    }
    private boolean validateForm() {
        if (nameField.getText().isBlank()) return false;
        if (sensorTypeCombo.getValue() == null) return false;
        if (virtualToggle.isSelected()) {
            if (virtualStrategyCombo.getValue() != null && virtualStrategyCombo.getValue().contains("Seno")) {
                // Validar campos seno
                try {
                    Double.parseDouble(sineOffsetField.getText());
                    Double.parseDouble(sineAmplitudeField.getText());
                    Double.parseDouble(sineFreqField.getText());
                } catch(Exception e) { return false; }
            } else {
                try { Double.parseDouble(baseValueField.getText()); } catch(Exception e) { return false; }
            }
        } else {
            if (webhookUrlField.getText().isBlank()) return false;
        }
        return true;
    }

    private void setupCombos() {
        sensorTypeCombo.setItems(FXCollections.observableArrayList(SensorType.getValidTypes()));
        sensorTypeCombo.setConverter(new StringConverter<>() {
            @Override public String toString(SensorType object) { return object == null ? "" : object.getFriendlyName(); }
            @Override public SensorType fromString(String string) { return null; }
        });
        virtualStrategyCombo.setItems(FXCollections.observableArrayList("Estocástica (Random Walk)", "Ondulatoria (Seno)"));
        virtualStrategyCombo.getSelectionModel().selectFirst();
        virtualStrategyCombo.valueProperty().addListener((obs, oldVal, newVal) -> {
            boolean isSine = newVal != null && newVal.contains("Seno");
            paramsSinePanel.setVisible(isSine); paramsSinePanel.setManaged(isSine);
            paramsRandomPanel.setVisible(!isSine); paramsRandomPanel.setManaged(!isSine);
        });
    }

    private void setupStrategySwitch() {
        strategyGroup.selectedToggleProperty().addListener((obs, oldVal, newVal) -> {
            if (newVal == null) return;
            boolean isVirtual = virtualToggle.isSelected();
            virtualPanel.setVisible(isVirtual); virtualPanel.setManaged(isVirtual);
            realPanel.setVisible(!isVirtual); realPanel.setManaged(!isVirtual);
        });
    }

    @FXML public void onCancel() { ((Stage)nameField.getScene().getWindow()).close(); }
    private void closeWindow() { ((Stage)nameField.getScene().getWindow()).close(); }
    private void showAlert(String t, String c) { new Alert(Alert.AlertType.WARNING, c).showAndWait(); }
}