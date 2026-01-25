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

    // Toggles (Now in Top Bar)
    @FXML
    private javafx.scene.control.ToggleButton viewHeatmapBtn;
    @FXML
    private javafx.scene.control.ToggleButton viewGraphBtn;
    @FXML
    private javafx.scene.control.ToggleGroup viewToggleGroup;

    // Sidebar Contexts
    @FXML
    private javafx.scene.layout.VBox sensorSidebar;
    @FXML
    private javafx.scene.layout.VBox graphSidebar;

    // Sidebar Content
    @FXML
    private FlowPane sensorsFlowPane;
    @FXML
    private Button addSensorBtn;

    @FXML
    private javafx.scene.control.TextArea cypherQueryInput;
    @FXML
    private javafx.scene.control.ListView<String> prebuiltQueriesList;

    // Managers
    private projectstalker.ui.view.dashboard.tabs.quality.KnowledgeGraphManager graphManager;

    // IMPORTANTE: Necesitamos saber en qué río estamos
    @Setter
    private TwinSummaryDTO currentTwinContext;

    private static final Map<String, String> PREBUILT_QUERIES = new java.util.LinkedHashMap<>();
    static {
        PREBUILT_QUERIES.put("Ver topología de sensores", "MATCH (a:Sensor)-[r]-(b:Sensor) RETURN a, r, b LIMIT 100");
        PREBUILT_QUERIES.put("Ver todo el grafo (Limit 30)", "MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 30"); // Updated
                                                                                                            // Limit
        PREBUILT_QUERIES.put("Nodos aislados", "MATCH (n) WHERE NOT (n)--() RETURN n LIMIT 50");

        // New Complex Query
        PREBUILT_QUERIES.put("Analizar Vertidos (Topológico/Espacial)",
                "MATCH (v:Vertido) " +
                        "CALL { " +
                        "    WITH v " +
                        "    OPTIONAL MATCH path_topo = shortestPath((v)-[:AGUAS_ABAJO|FLUYE_HACIA*..100]->(s_topo:Sensor)) "
                        +
                        "    RETURN s_topo, path_topo " +
                        "    LIMIT 1 " +
                        "} " +
                        "CALL { " +
                        "    WITH v " +
                        "    MATCH (s_prox:Sensor) " +
                        "    WITH s_prox, (v.utm_x - s_prox.utm_x)^2 + (v.utm_y - s_prox.utm_y)^2 as dist_sq " +
                        "    ORDER BY dist_sq ASC " +
                        "    LIMIT 1 " +
                        "    RETURN s_prox " +
                        "} " +
                        "WITH v, coalesce(s_topo, s_prox) as s_final, path_topo " +
                        "OPTIONAL MATCH (s_final)-[r_obj]-(obj:ObjetivoPrioritario) " +
                        "RETURN v, s_final, path_topo, r_obj, obj LIMIT 300");
    }

    @FXML
    public void initialize() {
        checkPermissions();
        loadHeatmapPlaceholder();

        // Init Graph Manager
        this.graphManager = new projectstalker.ui.view.dashboard.tabs.quality.KnowledgeGraphManager(graphContainer);

        setupViewToggles();
        setupGraphSidebar();
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

    private void setupGraphSidebar() {
        if (prebuiltQueriesList != null) {
            prebuiltQueriesList.getItems().addAll(PREBUILT_QUERIES.keySet());

            prebuiltQueriesList.setOnMouseClicked(e -> {
                String selected = prebuiltQueriesList.getSelectionModel().getSelectedItem();
                if (selected != null) {
                    String query = PREBUILT_QUERIES.get(selected);
                    cypherQueryInput.setText(query);
                    onRunCypherQuery();
                }
            });
        }
    }

    private void showGraphView() {
        // Content
        heatmapContainer.setVisible(false);
        heatmapContainer.setManaged(false);
        graphContainer.setVisible(true);
        graphContainer.setManaged(true);

        // Sidebar
        sensorSidebar.setVisible(false);
        sensorSidebar.setManaged(false);
        graphSidebar.setVisible(true);
        graphSidebar.setManaged(true);

        // Lazy Connect (Default Query if empty)
        if (cypherQueryInput.getText().trim().isEmpty()) {
            cypherQueryInput.setText(PREBUILT_QUERIES.get("Ver todo el grafo (Limit 300)"));
        }

        // Just ensure connection, don't run query automatically unless we want to
        // But for UX, let's run what's in the box if we just switched?
        // Or wait for user. Let's wait, but connect.
        this.graphManager.connect("bolt://localhost:7687", "neo4j", "12345678secret");
    }

    private void showHeatmapView() {
        // Content
        graphContainer.setVisible(false);
        graphContainer.setManaged(false);
        heatmapContainer.setVisible(true);
        heatmapContainer.setManaged(true);

        // Sidebar
        graphSidebar.setVisible(false);
        graphSidebar.setManaged(false);
        sensorSidebar.setVisible(true);
        sensorSidebar.setManaged(true);

        this.graphManager.stop();
    }

    @FXML
    public void onRunCypherQuery() {
        String query = cypherQueryInput.getText();
        if (query == null || query.isBlank())
            return;

        log.info("Running Cypher: {}", query);
        // We need to pass the query to manager
        this.graphManager.fetchGraph(query);
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