package projectstalker.ui.view.dashboard.tabs;

import javafx.application.Platform;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.layout.FlowPane;
import javafx.scene.layout.StackPane;
import javafx.stage.Modality;
import javafx.stage.Stage;
import javafx.stage.StageStyle;
import lombok.RequiredArgsConstructor;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.ApplicationContext;
import org.springframework.stereotype.Component;
import projectstalker.domain.dto.twin.TwinSummaryDTO;
import projectstalker.ui.security.AuthenticationService;
import projectstalker.ui.view.components.SensorWizardController;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

@Slf4j
@Component
@RequiredArgsConstructor
public class QualityTabController {

    private final AuthenticationService authService;
    private final projectstalker.ui.service.SensorClientService sensorService; // Injected
    private final ApplicationContext springContext; // Necesario para inyectar el Wizard

    private final org.springframework.context.ApplicationEventPublisher eventPublisher;

    @FXML
    private StackPane heatmapContainer;
    @FXML
    private StackPane graphContainer;
    @FXML
    private javafx.scene.control.ToggleButton viewHeatmapBtn;
    @FXML
    private javafx.scene.control.ToggleButton viewGraphBtn;
    @FXML
    private javafx.scene.control.ToggleGroup viewToggleGroup;
    @FXML
    private FlowPane sensorsFlowPane;
    @FXML
    private Button addSensorBtn;

    // Managers
    private projectstalker.ui.view.dashboard.tabs.quality.KnowledgeGraphManager graphManager;

    // IMPORTANTE: Necesitamos saber en qué río estamos
    @Setter
    private TwinSummaryDTO currentTwinContext;

    @FXML
    public void initialize() {
        checkPermissions();
        loadHeatmapPlaceholder();

        // Init Graph Manager
        this.graphManager = new projectstalker.ui.view.dashboard.tabs.quality.KnowledgeGraphManager(graphContainer);

        setupViewToggles();
    }

    private void setupViewToggles() {
        if (viewToggleGroup != null) {
            viewToggleGroup.selectedToggleProperty().addListener((obs, old, newVal) -> {
                if (newVal != null) {
                    if (newVal == viewGraphBtn) {
                        showGraphView();
                    } else {
                        showHeatmapView();
                    }
                }
            });
        }
    }

    private void showGraphView() {
        heatmapContainer.setVisible(false);
        heatmapContainer.setManaged(false);
        graphContainer.setVisible(true);
        graphContainer.setManaged(true);

        // Lazy Connect
        this.graphManager.connect("bolt://localhost:7687", "neo4j", "12345678secret");
    }

    private void showHeatmapView() {
        graphContainer.setVisible(false);
        graphContainer.setManaged(false);
        heatmapContainer.setVisible(true);
        heatmapContainer.setManaged(true);

        this.graphManager.stop();
    }

    // Método llamado por TwinDashboardController al cargar el tab
    public void setTwinContext(TwinSummaryDTO twinId) {
        this.currentTwinContext = twinId;
        loadSensorsList();
    }

    private void checkPermissions() {
        boolean canEdit = authService.getCurrentRoles().contains("ADMIN") ||
                authService.getCurrentRoles().contains("ANALYST");
        if (!canEdit) {
            addSensorBtn.setDisable(true);
            addSensorBtn.setVisible(false);
        }
    }

    @FXML
    public void onAddSensor() {
        // Fail Fast en el controlador padre también
        if (currentTwinContext == null) {
            log.error("Intento de añadir sensor sin contexto de río cargado.");
            return;
        }

        try {
            FXMLLoader loader = new FXMLLoader(getClass().getResource("/fxml/components/sensor-wizard.fxml"));
            loader.setControllerFactory(springContext::getBean);
            Parent root = loader.load();

            SensorWizardController wizard = loader.getController();
            wizard.setTwinContext(currentTwinContext);

            wizard.setOnSuccess(dto -> {
                log.info("Sensor creado! Recargando lista...");
                loadSensorsList();
                eventPublisher.publishEvent(new projectstalker.ui.event.SensorListRefreshEvent(this));
            });

            Stage stage = new Stage();
            stage.initModality(Modality.APPLICATION_MODAL);
            stage.setTitle("Nuevo Sensor - " + currentTwinContext.name());
            stage.setScene(new Scene(root, 650, 600));
            stage.showAndWait();

        } catch (IOException e) {
            log.error("Error UI al abrir wizard", e);
        } catch (IllegalArgumentException | IllegalStateException e) {
            // Capturamos el Fail Fast del Wizard
            log.error("Error de contexto al abrir wizard: {}", e.getMessage());
            Alert alert = new Alert(Alert.AlertType.ERROR, "Error de sistema: " + e.getMessage());
            alert.showAndWait();
        }
    }

    private void loadSensorsList() {
        if (currentTwinContext == null)
            return;

        sensorsFlowPane.getChildren().clear();
        log.info("Cargando lista de sensores para Twin: {}", currentTwinContext.id());

        // FIX: Now calling the real backend endpoint
        sensorService.getSensorsByTwin(currentTwinContext.id())
                .subscribe(sensor -> {
                    Platform.runLater(() -> {
                        sensorsFlowPane.getChildren().add(createSensorCard(sensor));
                    });
                }, err -> {
                    log.error("Error cargando lista de sensores", err);
                });
    }

    private javafx.scene.layout.Region createSensorCard(projectstalker.domain.dto.sensor.SensorResponseDTO sensor) {
        javafx.scene.layout.VBox card = new javafx.scene.layout.VBox(5);
        card.setStyle(
                "-fx-background-color: -color-bg-subtle; -fx-padding: 10; -fx-background-radius: 8; -fx-effect: dropshadow(three-pass-box, rgba(0,0,0,0.2), 5, 0, 0, 2);");
        card.setPrefWidth(180);

        javafx.scene.control.Label name = new javafx.scene.control.Label(sensor.name());
        name.setStyle("-fx-font-weight: bold; -fx-text-fill: -color-fg-default;");

        javafx.scene.control.Label type = new javafx.scene.control.Label(sensor.signalType());
        type.setStyle("-fx-font-size: 11px; -fx-text-fill: -color-fg-muted;");

        javafx.scene.layout.HBox valueBox = new javafx.scene.layout.HBox(5);
        valueBox.setAlignment(javafx.geometry.Pos.CENTER_LEFT);

        javafx.scene.control.Label valueLabel = new javafx.scene.control.Label("--");
        valueLabel.setStyle("-fx-font-size: 14px; -fx-font-weight: bold; -fx-text-fill: -color-accent-fg;");

        javafx.scene.control.Label unitLabel = new javafx.scene.control.Label(sensor.unit());
        unitLabel.setStyle("-fx-font-size: 11px; -fx-text-fill: -color-fg-muted;");

        valueBox.getChildren().addAll(valueLabel, unitLabel);

        // CLICK TO EDIT (Single Click as requested)
        card.setOnMouseClicked(e -> {
            openEditWizard(sensor);
        });
        // Cursor hand
        card.setStyle(card.getStyle() + "-fx-cursor: hand;");

        // Polling for live data
        startLivePolling(sensor.stationId(), sensor.name(), valueLabel);

        card.getChildren().addAll(name, type, new javafx.scene.control.Separator(), valueBox);
        return card;
    }

    private void openEditWizard(projectstalker.domain.dto.sensor.SensorResponseDTO sensor) {
        try {
            FXMLLoader loader = new FXMLLoader(getClass().getResource("/fxml/components/sensor-wizard.fxml"));
            loader.setControllerFactory(springContext::getBean);
            Parent root = loader.load();

            SensorWizardController wizard = loader.getController();
            wizard.setTwinContext(currentTwinContext);
            wizard.setSensorToEdit(sensor); // PRE-FILL

            wizard.setOnSuccess(dto -> {
                log.info("Sensor editado! Recargando lista...");
                loadSensorsList();
                eventPublisher.publishEvent(new projectstalker.ui.event.SensorListRefreshEvent(this));
            });

            Stage stage = new Stage();
            stage.initModality(Modality.APPLICATION_MODAL);
            stage.setTitle("Editar Sensor - " + sensor.name());
            stage.setScene(new Scene(root, 650, 600));
            stage.showAndWait();

        } catch (IOException e) {
            log.error("Error UI al abrir wizard de edición", e);
        }
    }

    private Map<String, javafx.animation.Timeline> activePolls = new HashMap<>();

    // Stop polling when view changes? Ideally yes. For now, we clear on
    // loadSensorsList maybe?
    // But loadSensorsList clears children only. Timelines stick around.
    // We should enable/disable polls?
    // For this task, let's just create them.

    private void startLivePolling(String stationId, String paramName, javafx.scene.control.Label target) {
        // Stop existing if any (key by ID)
        if (activePolls.containsKey(stationId)) {
            activePolls.get(stationId).stop();
        }

        javafx.animation.Timeline timeline = new javafx.animation.Timeline(
                new javafx.animation.KeyFrame(javafx.util.Duration.seconds(4), ev -> {
                    // Fetch Realtime
                    sensorService.getRealtime(stationId, "ALL")
                            .collectList()
                            .subscribe(readings -> {
                                // Assuming we get a list, take first
                                Platform.runLater(() -> {
                                    if (!readings.isEmpty()) {
                                        target.setText(readings.get(0).formattedValue());
                                    }
                                });
                            }, err -> log.debug("Poll error: " + err.getMessage()));
                }));
        timeline.setCycleCount(javafx.animation.Animation.INDEFINITE);
        timeline.play();
        activePolls.put(stationId, timeline);
    }

    private void loadHeatmapPlaceholder() {
        // STUB
    }
}