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
    @FXML
    private DatePicker startDatePicker;
    @FXML
    private DatePicker endDatePicker;

    // Pagination
    @FXML
    private Button prevPageBtn;
    @FXML
    private Button nextPageBtn;
    @FXML
    private Label pageLabel;
    @FXML
    private Label totalLabel;

    private int currentPage = 0;
    private int pageSize = 10;
    private int totalPages = 0;

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

        // Date Defaults (This Week)
        startDatePicker.setValue(java.time.LocalDate.now().minusMonths(1));
        endDatePicker.setValue(java.time.LocalDate.now());

        startDatePicker.valueProperty().addListener((o, old, newVal) -> {
            currentPage = 0;
            loadAlerts();
        });
        endDatePicker.valueProperty().addListener((o, old, newVal) -> {
            currentPage = 0;
            loadAlerts();
        });

        loadAlerts();

        // Multi-select persistence
        alertsTable.getSelectionModel().setSelectionMode(SelectionMode.MULTIPLE);

        // Auto-refresh every 60 seconds
        reactor.core.publisher.Flux.interval(java.time.Duration.ofSeconds(60))
                .subscribe(tick -> {
                    // Only refresh if on first page to avoid jumping around while browsing history
                    if (currentPage == 0) {
                        javafx.application.Platform.runLater(this::loadAlerts);
                    }
                });
    }

    @FXML
    public void onPrevPage() {
        if (currentPage > 0) {
            currentPage--;
            loadAlerts();
        }
    }

    @FXML
    public void onNextPage() {
        if (currentPage < totalPages - 1) {
            currentPage++;
            loadAlerts();
        }
    }

    private void updateMasterListSafely(java.util.List<AlertDTO> newList) {
        if (newList == null) {
            log.warn("updateMasterListSafely received null list");
            newList = java.util.Collections.emptyList();
        }
        log.info("Received {} alerts for current page", newList.size());
        masterList.setAll(newList);
        // Re-apply filter
        updateFilter(statusFilterCombo.getValue());
    }

    private void updateControls(projectstalker.ui.model.RestPage<AlertDTO> page) {
        this.totalPages = page.getTotalPages();
        this.totalLabel.setText("Total: " + page.getTotalElements());
        this.pageLabel.setText("Página " + (currentPage + 1) + " / " + (totalPages == 0 ? 1 : totalPages));

        this.prevPageBtn.setDisable(currentPage == 0);
        this.nextPageBtn.setDisable(currentPage >= totalPages - 1);
    }

    private void updateFilter(String statusLabel) {
        filteredList.setPredicate(alert -> {
            if (statusLabel == null || "TODOS".equals(statusLabel))
                return true;

            boolean visible = switch (statusLabel) {
                case "NUEVO/ACTIVO" -> alert.status() == AlertStatus.NEW || alert.status() == AlertStatus.ACTIVE;
                case "RECONOCIDO" -> alert.status() == AlertStatus.ACKNOWLEDGED;
                case "RESUELTO" -> alert.status() == AlertStatus.RESOLVED;
                default -> true;
            };
            return visible;
        });
        log.debug("List size after filter '{}': {}", statusLabel, filteredList.size());
        updateBulkButtons();
    }

    private void setupTable() {
        alertsTable.setItems(filteredList);

        // Double Click Listener
        alertsTable.setRowFactory(tv -> {
            TableRow<AlertDTO> row = new TableRow<>();
            row.setOnMouseClicked(event -> {
                if (event.getClickCount() == 2 && (!row.isEmpty())) {
                    AlertDTO rowData = row.getItem();
                    handleDoubleClick(rowData);
                }
            });
            return row;
        });

        alertsTable.getSelectionModel().selectedItemProperty().addListener((obs, old, newVal) -> updateBulkButtons());
        alertsTable.getSelectionModel().getSelectedItems().addListener(
                (javafx.collections.ListChangeListener.Change<? extends AlertDTO> c) -> updateBulkButtons());

        // Columns setup
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

        // Renderers
        setupRenderers();
    }

    private void handleDoubleClick(AlertDTO alert) {
        if (alert.reportId() != null) {
            log.info("Opening report for alert {}", alert.id());
            // Fetch Report
            alertService.getReport(alert.reportId())
                    .subscribe(report -> {
                        javafx.application.Platform.runLater(() -> showReportDialog(report));
                    }, err -> showErr("Error al abrir informe", err));
        } else if (alert.status() == AlertStatus.RESOLVED) {
            Alert info = new Alert(Alert.AlertType.INFORMATION);
            info.setTitle("Alerta Resuelta");
            info.setHeaderText("Esta alerta está resuelta pero no tiene informe adjunto.");
            info.show();
        }
    }

    private void showReportDialog(projectstalker.domain.dto.report.ReportDTO report) {
        Dialog<Void> dialog = new Dialog<>();
        dialog.setTitle("Detalles del Informe");
        dialog.setHeaderText(report.title());

        ButtonType closeBtn = new ButtonType("Cerrar", ButtonBar.ButtonData.OK_DONE);
        dialog.getDialogPane().getButtonTypes().add(closeBtn);

        TextArea body = new TextArea(report.body());
        body.setEditable(false);
        body.setWrapText(true);

        javafx.scene.layout.VBox content = new javafx.scene.layout.VBox(10);
        content.getChildren().add(new Label("Creado el: " + report.createdAt()));
        content.getChildren().add(body);

        dialog.getDialogPane().setContent(content);
        dialog.show();
    }

    private void setupRenderers() {
        // Gravedad
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
                    if (item == AlertSeverity.CRITICAL || item == AlertSeverity.WARNING || item == AlertSeverity.INFO)
                        label.setGraphic(icon);
                    setGraphic(label);
                }
            }
        });

        // Fecha
        DateTimeFormatter dtf = DateTimeFormatter.ofPattern("dd/MM HH:mm:ss");
        colTime.setCellFactory(col -> new TableCell<>() {
            @Override
            protected void updateItem(LocalDateTime item, boolean empty) {
                super.updateItem(item, empty);
                if (empty || item == null)
                    setText(null);
                else
                    setText(item.format(dtf));
            }
        });

        // Actions
        colActions.setCellFactory(col -> new TableCell<>() {
            private final Button btn = new Button();

            @Override
            protected void updateItem(Void item, boolean empty) {
                super.updateItem(item, empty);
                AlertDTO alert = getTableRow().getItem();
                if (empty || alert == null) {
                    setGraphic(null);
                    return;
                }
                if (alert.status() == AlertStatus.NEW) {
                    btn.setText("ACK");
                    btn.getStyleClass().setAll("button", "btn-xs", "btn-success");
                    btn.setStyle("-fx-background-color: #4a90e2; -fx-text-fill: white;");
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
        boolean hasAck = selected.stream().anyMatch(a -> a.status() == AlertStatus.ACKNOWLEDGED); // Fixed logic: only
                                                                                                  // enable if we have
                                                                                                  // ACK items for
                                                                                                  // Resolve

        bulkAckBtn.setDisable(!hasNew);
        bulkResolveBtn.setDisable(!hasAck);
        bulkAckBtn.setText("Confirmar (" + selected.stream().filter(a -> a.status() == AlertStatus.NEW).count() + ")");
        bulkResolveBtn.setText(
                "Resolver (" + selected.stream().filter(a -> a.status() == AlertStatus.ACKNOWLEDGED).count() + ")");
    }

    private void handleAck(AlertDTO alert) {
        alertService.acknowledgeAlert(alert.id().toString())
                .delayElement(java.time.Duration.ofMillis(200))
                .subscribe(v -> loadAlerts(), err -> showErr("Error ACK", err)); // loadAlerts will refresh current page
    }

    private void handleResolve(AlertDTO alert) {
        // Reuse Bulk Logic
        showResolveDialog(List.of(alert));
    }

    @FXML
    public void onBulkAck() {
        var selected = alertsTable.getSelectionModel().getSelectedItems().stream()
                .filter(a -> a.status() == AlertStatus.NEW).toList();
        if (selected.isEmpty())
            return;
        reactor.core.publisher.Flux.fromIterable(selected)
                .flatMap(a -> alertService.acknowledgeAlert(a.id().toString()))
                .collectList().subscribe(v -> loadAlerts(), err -> showErr("Error Bulk ACK", err));
    }

    @FXML
    public void onBulkResolve() {
        var selected = alertsTable.getSelectionModel().getSelectedItems().stream()
                .filter(a -> a.status() == AlertStatus.ACKNOWLEDGED).toList();
        if (selected.isEmpty())
            return;
        showResolveDialog(selected);
    }

    private void showResolveDialog(List<AlertDTO> selected) {
        Dialog<javafx.util.Pair<String, String>> dialog = new Dialog<>();
        dialog.setTitle("Crear Informe y Resolver");
        dialog.setHeaderText("Resolver " + selected.size() + " alerta(s)");
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
            if (dialogButton == loginButtonType)
                return new javafx.util.Pair<>(title.getText(), body.getText());
            return null;
        });
        dialog.showAndWait().ifPresent(reportDetails -> {
            List<String> ids = selected.stream().map(AlertDTO::id).collect(java.util.stream.Collectors.toList());
            var req = new projectstalker.domain.dto.report.CreateReportRequest(
                    reportDetails.getKey(), reportDetails.getValue(), ids);
            alertService.createReport(req).delayElement(java.time.Duration.ofMillis(200))
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
        LocalDateTime start = startDatePicker.getValue().atStartOfDay();
        LocalDateTime end = endDatePicker.getValue().atTime(23, 59, 59);

        alertService.getAlerts(currentPage, pageSize, start, end)
                .subscribe(
                        page -> javafx.application.Platform.runLater(() -> {
                            updateMasterListSafely(page.getContent());
                            updateControls(page);
                        }),
                        err -> log.error("Error cargando alertas", err));
    }

    @FXML
    public void onConfigureRules() {
        try {
            FXMLLoader loader = new FXMLLoader(getClass().getResource("/fxml/components/rule-config-dialog.fxml"));
            loader.setControllerFactory(springContext::getBean);
            javafx.scene.Parent root = loader.load();
            projectstalker.ui.view.components.RuleConfigDialogController controller = loader.getController();
            controller.setOnSaveCallback(this::loadAlerts);
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
        if (filteredList.isEmpty()) {
            warn("No hay datos para exportar.");
            return;
        }

        javafx.stage.FileChooser fileChooser = new javafx.stage.FileChooser();
        fileChooser.setTitle("Guardar Log de Alertas");
        fileChooser.getExtensionFilters().add(new javafx.stage.FileChooser.ExtensionFilter("CSV Files", "*.csv"));
        fileChooser.setInitialFileName("alertas_"
                + java.time.LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmm")) + ".csv");

        java.io.File file = fileChooser.showSaveDialog(alertsTable.getScene().getWindow());
        if (file != null) {
            try (java.io.PrintWriter writer = new java.io.PrintWriter(file, java.nio.charset.StandardCharsets.UTF_8)) {
                // Header
                writer.println("Fecha,Gravedad,Sensor,Mensaje,Estado,Valor,Métrica");

                // Content (Using filtered list to export exactly what user sees/filters)
                for (AlertDTO alert : filteredList) {
                    String line = String.format("%s,%s,%s,\"%s\",%s,%s,%s",
                            alert.timestamp(),
                            alert.severity(),
                            alert.stationName(),
                            alert.message().replace("\"", "\"\""), // Escape quotes
                            alert.status(),
                            alert.value(),
                            alert.metric());
                    writer.println(line);
                }

                Alert info = new Alert(Alert.AlertType.INFORMATION);
                info.setTitle("Exportación Exitosa");
                info.setHeaderText(null);
                info.setContentText("Log exportado correctamente a:\n" + file.getAbsolutePath());
                info.show();

            } catch (Exception e) {
                showErr("Error al exportar log", e);
            }
        }
    }

    private void warn(String msg) {
        Alert alert = new Alert(Alert.AlertType.WARNING);
        alert.setTitle("Aviso");
        alert.setHeaderText(null);
        alert.setContentText(msg);
        alert.show();
    }
}