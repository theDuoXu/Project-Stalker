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

import java.util.List;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

@Slf4j
@Component
@RequiredArgsConstructor
public class AlertsTabController {

    private final AlertClientService alertService;
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
    private TableColumn<AlertDTO, Void> colActions;

    @FXML
    private Button bulkAckBtn;
    @FXML
    private Button bulkResolveBtn;
    @FXML
    private ComboBox<String> statusFilterCombo;

    private final ObservableList<AlertDTO> masterList = FXCollections.observableArrayList();
    private final javafx.collections.transformation.FilteredList<AlertDTO> filteredList = new javafx.collections.transformation.FilteredList<>(
            masterList, p -> true);

    @FXML
    public void initialize() {
        setupTable();

        // Status Filter Setup
        ObservableList<String> validStatuses = FXCollections.observableArrayList(
                "TODOS", "NUEVO/ACTIVO", "RECONOCIDO", "RESUELTO");
        statusFilterCombo.setItems(validStatuses);
        statusFilterCombo.getSelectionModel().select("TODOS");
        statusFilterCombo.valueProperty().addListener((obs, oldVal, newVal) -> updateFilter(newVal));

        loadAlerts();

        // Multi-select persistence
        alertsTable.getSelectionModel().setSelectionMode(SelectionMode.MULTIPLE);

        // Auto-refresh every 30 seconds
        reactor.core.publisher.Flux.interval(java.time.Duration.ofSeconds(30))
                .flatMap(tick -> alertService.getAllAlerts().collectList()) // Fetch ALL to keep RESOLVED items visible
                .subscribe(
                        newList -> javafx.application.Platform.runLater(() -> {
                            updateMasterListSafely(newList);
                            log.debug("Alertas actualizadas (Auto-Refresh)");
                        }),
                        err -> log.error("Error en auto-refresh de alertas", err));
    }

    private void updateMasterListSafely(java.util.List<AlertDTO> newList) {
        // Save selection
        var selectedIds = alertsTable.getSelectionModel().getSelectedItems().stream()
                .map(AlertDTO::id)
                .collect(java.util.stream.Collectors.toSet());

        masterList.setAll(newList);

        // Restore selection (tricky with FilteredList, but we try)
        // We need to find the objects in the FilteredList that match the IDs
        for (AlertDTO item : alertsTable.getItems()) {
            if (selectedIds.contains(item.id())) {
                alertsTable.getSelectionModel().select(item);
            }
        }
    }

    private void updateFilter(String statusLabel) {
        filteredList.setPredicate(alert -> {
            if (statusLabel == null || "TODOS".equals(statusLabel))
                return true;

            return switch (statusLabel) {
                case "NUEVO/ACTIVO" -> alert.status() == AlertStatus.NEW || alert.status() == AlertStatus.ACTIVE;
                case "RECONOCIDO" -> alert.status() == AlertStatus.ACKNOWLEDGED;
                case "RESUELTO" -> alert.status() == AlertStatus.RESOLVED;
                default -> true;
            };
        });
        updateBulkButtons(); // Refresh buttons based on visible selection
    }

    private void setupTable() {
        alertsTable.setItems(filteredList); // Use FilteredList

        alertsTable.getSelectionModel().selectedItemProperty().addListener((obs, old, newVal) -> {
            updateBulkButtons();
        });

        alertsTable.getSelectionModel().getSelectedItems()
                .addListener((javafx.collections.ListChangeListener.Change<? extends AlertDTO> c) -> {
                    updateBulkButtons();
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
                        default -> {
                            // nothing
                        }
                    }
                    if (item == AlertSeverity.CRITICAL || item == AlertSeverity.WARNING || item == AlertSeverity.INFO) {
                        label.setGraphic(icon);
                    }
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

        // ACTION BUTTONS
        colActions.setCellFactory(col -> new TableCell<>() {
            private final Button btn = new Button();

            @Override
            protected void updateItem(Void item, boolean empty) {
                super.updateItem(item, empty);
                AlertDTO alert = getTableRow().getItem(); // Safe way to get item

                // If the row is empty or alert is null, clear graphic
                if (empty || alert == null) {
                    setGraphic(null);
                    return;
                }

                // Decide button based on status
                if (alert.status() == AlertStatus.NEW) {
                    btn.setText("ACK");
                    btn.getStyleClass().setAll("button", "btn-xs", "btn-success");
                    btn.setStyle("-fx-background-color: #4a90e2; -fx-text-fill: white;"); // Inline override if css
                                                                                          // fails
                    btn.setOnAction(e -> handleAck(alert));
                    setGraphic(btn);
                } else if (alert.status() == AlertStatus.ACKNOWLEDGED) {
                    btn.setText("Resolver");
                    btn.getStyleClass().setAll("button", "btn-xs", "btn-info");
                    btn.setStyle("-fx-background-color: #50c878; -fx-text-fill: white;");
                    btn.setOnAction(e -> handleResolve(alert));
                    setGraphic(btn);
                } else {
                    setGraphic(null);
                }
            }
        });
    }

    private void updateBulkButtons() {
        if (bulkAckBtn == null || bulkResolveBtn == null)
            return;

        var selected = alertsTable.getSelectionModel().getSelectedItems();
        boolean hasNew = selected.stream().anyMatch(a -> a.status() == AlertStatus.NEW);
        boolean hasAck = selected.stream().anyMatch(a -> a.status() == AlertStatus.ACKNOWLEDGED);

        bulkAckBtn.setDisable(!hasNew);
        bulkResolveBtn.setDisable(!hasAck);

        bulkAckBtn.setText("Confirmar (" + selected.stream().filter(a -> a.status() == AlertStatus.NEW).count() + ")");
        bulkResolveBtn.setText(
                "Resolver (" + selected.stream().filter(a -> a.status() == AlertStatus.ACKNOWLEDGED).count() + ")");
    }

    private void handleAck(AlertDTO alert) {
        log.info("Acknowledging alert {}", alert.id());
        alertService.acknowledgeAlert(alert.id().toString())
                .delayElement(java.time.Duration.ofMillis(200)) // Ensure DB commit
                .subscribe(v -> {
                    log.info("ACK SUCCESS for {}. Reloading...", alert.id());
                    loadAlerts();
                }, err -> showErr("Error ACK", err)); // loadAlerts handles threading now
    }

    private void handleResolve(AlertDTO alert) {
        log.info("Resolving alert {}", alert.id());

        // Reuse the logic from Bulk Resolve but for a single item
        // Custom Dialog for Report Details
        Dialog<javafx.util.Pair<String, String>> dialog = new Dialog<>();
        dialog.setTitle("Crear Informe y Resolver");
        dialog.setHeaderText("Resolver alerta: " + alert.metric());

        ButtonType loginButtonType = new ButtonType("Resolver", ButtonBar.ButtonData.OK_DONE);
        dialog.getDialogPane().getButtonTypes().addAll(loginButtonType, ButtonType.CANCEL);

        javafx.scene.layout.GridPane grid = new javafx.scene.layout.GridPane();
        grid.setHgap(10);
        grid.setVgap(10);
        grid.setPadding(new javafx.geometry.Insets(20, 150, 10, 10));

        TextField title = new TextField();
        title.setPromptText("Título del informe");
        TextArea body = new TextArea();
        body.setPromptText("Detalles de la incidencia...");

        grid.add(new Label("Título:"), 0, 0);
        grid.add(title, 1, 0);
        grid.add(new Label("Detalles:"), 0, 1);
        grid.add(body, 1, 1);

        dialog.getDialogPane().setContent(grid);

        dialog.setResultConverter(dialogButton -> {
            if (dialogButton == loginButtonType) {
                return new javafx.util.Pair<>(title.getText(), body.getText());
            }
            return null;
        });

        java.util.Optional<javafx.util.Pair<String, String>> result = dialog.showAndWait();

        result.ifPresent(reportDetails -> {
            var req = new projectstalker.domain.dto.report.CreateReportRequest(
                    reportDetails.getKey(),
                    reportDetails.getValue(),
                    List.of(alert.id().toString()));

            alertService.createReport(req)
                    .delayElement(java.time.Duration.ofMillis(200))
                    .subscribe(v -> {
                        log.info("RESOLVE SUCCESS for {}. Reloading...", alert.id());
                        loadAlerts();
                    }, err -> showErr("Error Reporte", err));
        });
    }

    @FXML
    public void onBulkAck() {
        var selected = alertsTable.getSelectionModel().getSelectedItems().stream()
                .filter(a -> a.status() == AlertStatus.NEW)
                .toList();

        if (selected.isEmpty())
            return;

        reactor.core.publisher.Flux.fromIterable(selected)
                .flatMap(a -> alertService.acknowledgeAlert(a.id().toString()))
                .collectList()
                .subscribe(v -> loadAlerts(), err -> showErr("Error Bulk ACK", err));
    }

    @FXML
    public void onBulkResolve() {
        var selected = alertsTable.getSelectionModel().getSelectedItems().stream()
                .filter(a -> a.status() == AlertStatus.ACKNOWLEDGED)
                .toList();

        if (selected.isEmpty()) {
            log.warn("Bulk Resolve: No ACKNOWLEDGED alerts selected. Total selection size: {}",
                    alertsTable.getSelectionModel().getSelectedItems().size());
            return;
        }

        log.info("Bulk Resolve initiated for {} items", selected.size());

        // Custom Dialog for Report Details
        Dialog<javafx.util.Pair<String, String>> dialog = new Dialog<>();
        dialog.setTitle("Crear Informe y Resolver");
        dialog.setHeaderText("Resolver " + selected.size() + " alertas seleccionadas");

        ButtonType loginButtonType = new ButtonType("Resolver", ButtonBar.ButtonData.OK_DONE);
        dialog.getDialogPane().getButtonTypes().addAll(loginButtonType, ButtonType.CANCEL);

        javafx.scene.layout.GridPane grid = new javafx.scene.layout.GridPane();
        grid.setHgap(10);
        grid.setVgap(10);
        grid.setPadding(new javafx.geometry.Insets(20, 150, 10, 10));

        TextField title = new TextField();
        title.setPromptText("Título del informe");
        TextArea body = new TextArea();
        body.setPromptText("Detalles de la incidencia...");

        grid.add(new Label("Título:"), 0, 0);
        grid.add(title, 1, 0);
        grid.add(new Label("Detalles:"), 0, 1);
        grid.add(body, 1, 1);

        dialog.getDialogPane().setContent(grid);

        // Convert the result to a title-body-pair when the login button is clicked.
        dialog.setResultConverter(dialogButton -> {
            if (dialogButton == loginButtonType) {
                return new javafx.util.Pair<>(title.getText(), body.getText());
            }
            return null;
        });

        java.util.Optional<javafx.util.Pair<String, String>> result = dialog.showAndWait();

        result.ifPresent(reportDetails -> {
            List<String> ids = selected.stream().map(AlertDTO::id).collect(java.util.stream.Collectors.toList());
            var req = new projectstalker.domain.dto.report.CreateReportRequest(
                    reportDetails.getKey(),
                    reportDetails.getValue(),
                    ids);

            alertService.createReport(req)
                    .delayElement(java.time.Duration.ofMillis(200))
                    .subscribe(v -> loadAlerts(), err -> showErr("Error Reporte", err));
        });
    }

    private void showErr(String title, Throwable e) {
        log.error(title, e);
        javafx.application.Platform.runLater(() -> {
            Alert err = new Alert(Alert.AlertType.ERROR);
            err.setTitle("Error");
            err.setHeaderText(title);
            err.setContentText(e.getMessage());
            err.show();
        });
    }

    private void loadAlerts() {
        // Use getAllAlerts so we can filter locally for RESOLVED items
        alertService.getAllAlerts()
                .collectList()
                .subscribe(
                        newList -> javafx.application.Platform.runLater(() -> updateMasterListSafely(newList)),
                        err -> log.error("Error cargando alertas", err));
    }

    @FXML
    public void onConfigureRules() {
        try {
            FXMLLoader loader = new FXMLLoader(getClass().getResource("/fxml/components/rule-config-dialog.fxml"));
            loader.setControllerFactory(springContext::getBean);
            javafx.scene.Parent root = loader.load();

            // Inject Refresh Callback
            projectstalker.ui.view.components.RuleConfigDialogController controller = loader.getController();
            controller.setOnSaveCallback(() -> {
                log.info("Rule configuration saved. Refreshing alerts...");
                loadAlerts(); // Reload manually to be sure
            });

            javafx.stage.Stage stage = new javafx.stage.Stage();
            stage.initModality(javafx.stage.Modality.APPLICATION_MODAL);
            stage.setTitle("Configurar Reglas de Alerta");
            stage.setScene(new javafx.scene.Scene(root));
            stage.setMinWidth(900);
            stage.setMinHeight(600);
            stage.showAndWait();
        } catch (java.io.IOException e) {
            log.error("Error opening rule config dialog", e);
        }
    }

    @FXML
    public void onExportLog() {
        log.info("[STUB] Exportando log...");
        Alert info = new Alert(Alert.AlertType.INFORMATION);
        info.setTitle("Exportar Log");
        info.setHeaderText("Funcionalidad no implementada");
        info.setContentText("La exportación de logs se implementará en la próxima fase.");
        info.show();
    }

}