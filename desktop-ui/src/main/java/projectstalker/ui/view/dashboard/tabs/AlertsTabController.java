package projectstalker.ui.view.dashboard.tabs;

import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.control.*;
import javafx.scene.layout.HBox;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.kordamp.ikonli.javafx.FontIcon;
import org.springframework.context.ApplicationContext;
import org.springframework.stereotype.Component;
import projectstalker.domain.dto.alert.AlertDTO;
import projectstalker.domain.dto.alert.AlertSeverity;
import projectstalker.domain.dto.alert.AlertStatus;
import projectstalker.ui.service.AlertClientService;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

@Slf4j
@Component
@RequiredArgsConstructor
public class AlertsTabController {

    private final AlertClientService alertService;
    private final projectstalker.ui.service.SensorClientService sensorService;
    private final ApplicationContext springContext;

    @FXML
    private TableView<AlertDTO> alertsTable;
    @FXML
    private TableColumn<AlertDTO, AlertSeverity> colSeverity;
    @FXML
    private TableColumn<AlertDTO, LocalDateTime> colTime;
    @FXML
    private TableColumn<AlertDTO, String> colMessage;
    @FXML
    private TableColumn<AlertDTO, String> colSensor;
    @FXML
    private TableColumn<AlertDTO, AlertStatus> colStatus;
    @FXML
    private TableColumn<AlertDTO, Void> colActions; // Columna de botones

    private final ObservableList<AlertDTO> alertsList = FXCollections.observableArrayList();

    @FXML
    public void initialize() {
        setupTable();
        loadAlerts();

        // Auto-refresh every 30 seconds
        reactor.core.publisher.Flux.interval(java.time.Duration.ofSeconds(30))
                .flatMap(tick -> alertService.getActiveAlerts().collectList())
                .subscribe(
                        newList -> javafx.application.Platform.runLater(() -> {
                            alertsList.setAll(newList);
                            log.debug("Alertas actualizadas (Auto-Refresh)");
                        }),
                        err -> log.error("Error en auto-refresh de alertas", err));
    }

    private void setupTable() {
        alertsTable.setItems(alertsList);

        // Selection Listener for Chart
        alertsTable.getSelectionModel().selectedItemProperty().addListener((obs, old, newVal) -> {
            if (newVal != null) {
                openChartDialog(newVal);
            }
        });

        // 1. Configurar Columnas simples
        colSeverity.setCellValueFactory(
                data -> new javafx.beans.property.SimpleObjectProperty<>(data.getValue().severity()));
        colTime.setCellValueFactory(
                data -> new javafx.beans.property.SimpleObjectProperty<>(data.getValue().timestamp()));
        colMessage
                .setCellValueFactory(data -> new javafx.beans.property.SimpleStringProperty(data.getValue().message()));
        colSensor.setCellValueFactory(
                data -> new javafx.beans.property.SimpleStringProperty(data.getValue().stationName()));
        colStatus.setCellValueFactory(
                data -> new javafx.beans.property.SimpleObjectProperty<>(data.getValue().status()));

        // 2. Renderizado Personalizado (Cell Factories)

        // Gravedad con Iconos y Colores
        colSeverity.setCellFactory(col -> new TableCell<>() {
            @Override
            protected void updateItem(AlertSeverity item, boolean empty) {
                super.updateItem(item, empty);
                if (empty || item == null) {
                    setGraphic(null);
                    setText(null);
                } else {
                    Label label = new Label(item.name());
                    FontIcon icon = new FontIcon();

                    switch (item) {
                        case CRITICAL -> {
                            icon.setIconLiteral("mdi2a-alert-decagram");
                            icon.setIconColor(javafx.scene.paint.Color.RED);
                            label.setStyle("-fx-text-fill: red; -fx-font-weight: bold;");
                        }
                        case WARNING -> {
                            icon.setIconLiteral("mdi2a-alert");
                            icon.setIconColor(javafx.scene.paint.Color.ORANGE);
                            label.setStyle("-fx-text-fill: orange;");
                        }
                        case INFO -> {
                            icon.setIconLiteral("mdi2i-information");
                            icon.setIconColor(javafx.scene.paint.Color.LIGHTBLUE);
                            label.setStyle("-fx-text-fill: white;");
                        }
                    }
                    label.setGraphic(icon);
                    setGraphic(label);
                }
            }
        });

        // Fecha Formateada
        DateTimeFormatter dtf = DateTimeFormatter.ofPattern("dd/MM HH:mm:ss");
        colTime.setCellFactory(col -> new TableCell<>() {
            @Override
            protected void updateItem(LocalDateTime item, boolean empty) {
                super.updateItem(item, empty);
                if (empty || item == null) {
                    setText(null);
                } else {
                    setText(item.format(dtf));
                }
            }
        });

        // Acciones (Botón "Reconocer")
        colActions.setCellFactory(col -> new TableCell<>() {
            private final Button btnAck = new Button("Reconocer");
            {
                btnAck.getStyleClass().addAll("btn-xs", "btn-success");
                btnAck.setOnAction(evt -> {
                    AlertDTO alert = getTableView().getItems().get(getIndex());
                    acknowledge(alert);
                });
            }

            @Override
            protected void updateItem(Void item, boolean empty) {
                super.updateItem(item, empty);
                if (empty) {
                    setGraphic(null);
                } else {
                    AlertDTO alert = getTableView().getItems().get(getIndex());
                    // Solo mostrar si está activa
                    if (alert.status() == AlertStatus.ACTIVE || alert.status() == null) {
                        setGraphic(btnAck);
                    } else {
                        setGraphic(null);
                    }
                }
            }
        });
    }

    private void loadAlerts() {
        alertService.getActiveAlerts()
                .subscribe(
                        alertsList::add,
                        err -> log.error("Error cargando alertas", err));
    }

    private void acknowledge(AlertDTO alert) {
        log.info("Usuario reconoce alerta: {}", alert.id());
        alertService.acknowledgeAlert(alert.id())
                .subscribe(v -> {
                    // Refresh simple: Recargar todo (en prod optimizar)
                    alertsList.clear();
                    loadAlerts();
                });
    }

    @FXML
    public void onConfigureRules() {
        try {
            FXMLLoader loader = new FXMLLoader(getClass().getResource("/fxml/components/rule-config-dialog.fxml"));
            loader.setControllerFactory(springContext::getBean);
            javafx.scene.Parent root = loader.load();

            javafx.stage.Stage stage = new javafx.stage.Stage();
            stage.initModality(javafx.stage.Modality.APPLICATION_MODAL);
            stage.setTitle("Configurar Reglas de Alerta");
            stage.setScene(new javafx.scene.Scene(root));
            stage.showAndWait();
        } catch (java.io.IOException e) {
            log.error("Error opening rule config dialog", e);
        }
    }

    @FXML
    public void onExportLog() {
        log.info("[STUB] Exportando log...");
    }

    private void openChartDialog(AlertDTO alert) {
        if (alert.metric() == null || alert.stationId() == null)
            return;

        javafx.scene.chart.NumberAxis xAxis = new javafx.scene.chart.NumberAxis();
        xAxis.setLabel("Hora (Horas desde inicio)");
        javafx.scene.chart.NumberAxis yAxis = new javafx.scene.chart.NumberAxis();
        yAxis.setLabel(alert.metric());

        javafx.scene.chart.LineChart<Number, Number> lineChart = new javafx.scene.chart.LineChart<>(xAxis, yAxis);
        lineChart.setTitle("Evolución de " + alert.metric() + " - " + alert.stationName());

        javafx.scene.chart.XYChart.Series<Number, Number> series = new javafx.scene.chart.XYChart.Series<>();
        series.setName("Valores");

        // Fetch data: +/- 12 hours from alert timestamp
        LocalDateTime start = alert.timestamp().minusHours(12);
        LocalDateTime end = alert.timestamp().plusHours(12);

        // Fetch history (returns Mono<SensorResponseDTO>)
        sensorService.getHistory(alert.stationId(), alert.metric())
                .subscribe(response -> {
                    javafx.application.Platform.runLater(() -> {
                        if (response.values() != null) {
                            for (var r : response.values()) {
                                // Simple X: Calculate relative hours from alert time
                                // Parse string date from ReadingDTO
                                LocalDateTime rTime = LocalDateTime.parse(r.timestamp());

                                long minutesDiff = java.time.temporal.ChronoUnit.MINUTES.between(alert.timestamp(),
                                        rTime);
                                // Only show points within +/- 12 hours window
                                if (Math.abs(minutesDiff) < 720) {
                                    series.getData()
                                            .add(new javafx.scene.chart.XYChart.Data<>(minutesDiff / 60.0, r.value()));
                                }
                            }
                            lineChart.getData().add(series);
                        }
                    });
                }, err -> log.error("Error fetching history for chart", err));

        javafx.scene.Scene scene = new javafx.scene.Scene(lineChart, 800, 600);
        javafx.stage.Stage stage = new javafx.stage.Stage();
        stage.setTitle("Detalle de Alerta: " + alert.metric());
        stage.setScene(scene);
        stage.show();
    }
}