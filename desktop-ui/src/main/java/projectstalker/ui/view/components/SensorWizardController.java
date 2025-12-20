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
    // Spring busca todas las clases @Component que implementan la interfaz y las mete en la lista
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
    public void initialize() {
        setupCommonCombos();
        setupStrategyLogic();
        setupLocationLogic();
    }

    public void setTwinContext(TwinSummaryDTO twin) {
        Objects.requireNonNull(twin);
        this.twinContext = twin;
        double lengthKm = twin.totalLengthKm();
        double dxMeters = twin.totalLengthKm() / twin.cellCount();

        if (lengthKm <= 0 || dxMeters <= 0)
            throw new IllegalArgumentException("La longitud o la resolución espacial no pueden ser menor o igual a cero");

        this.spatialResolutionKm = dxMeters / 1000.0;
        locationSlider.setMax(lengthKm);
        locationSlider.setMajorTickUnit(Math.max(5.0, lengthKm / 10.0));
    }

    public void setOnSuccess(Consumer<SensorCreationDTO> cb) {
        this.onSuccessCallback = cb;
    }

    private void setupLocationLogic() {
        locationSlider.valueProperty().addListener((obs, old, val) -> {
            if (spatialResolutionKm <= 0) return;
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

    private void filterStrategies() {
        boolean isVirtual = virtualToggle.isSelected();
        List<SensorUiStrategy> filtered = availableStrategies.stream()
                .filter(s -> s.isVirtual() == isVirtual)
                .collect(Collectors.toList());

        strategySelector.setItems(FXCollections.observableArrayList(filtered));
        if (!filtered.isEmpty()) strategySelector.getSelectionModel().selectFirst();
    }

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
            sensorService.createSensor(dto)
                    .subscribe(r -> Platform.runLater(() -> {
                        if (onSuccessCallback != null) onSuccessCallback.accept(dto);
                        closeWindow();
                    }), e -> Platform.runLater(() -> {
                        saveButton.setDisable(false);
                        showAlert("Error Backend: " + e.getMessage());
                    }));

        } catch (Exception e) {
            log.error("Error", e);
        }
    }

    private SensorCreationDTO buildDto(SensorUiStrategy strategy) {
        // Extraemos configuración de la estrategia
        Map<String, Object> config = strategy.extractConfiguration();

        // Añadimos metadatos del contexto
        config.put("twinId", twinContext.id());
        config.put("strategy", strategy.getStrategyCode());

        return new SensorCreationDTO(
                nameField.getText(),
                sensorTypeCombo.getValue().getCode(),
                locationSlider.getValue(),
                strategy.isVirtual() ? "VIRTUAL" : "REAL",
                config
        );
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