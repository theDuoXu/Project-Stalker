package projectstalker.ui.view.dashboard.tabs;

import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;
import javafx.scene.control.*;
import javafx.scene.control.cell.PropertyValueFactory;
import javafx.scene.layout.HBox;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.kordamp.ikonli.javafx.FontIcon;
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
    }

    private void setupTable() {
        alertsTable.setItems(alertsList);

        // 1. Configurar Columnas simples
        // Nota: Las columnas deben tener fx:id en el FXML para que esto funcione
        // automáticamente
        // Si no, las buscamos por índice o las inyectamos.
        // Como el FXML actual no tiene fx:id en las columnas, las configuramos
        // programáticamente por índice es arriesgado
        // Mejor actualizaré el FXML primero. Pero por ahora, asumo que las inyecto si
        // coinciden los nombres.
        // Si fallan, usaré lookup.

        // Mapeo
        colSeverity.setCellValueFactory(new PropertyValueFactory<>("severity"));
        colTime.setCellValueFactory(new PropertyValueFactory<>("timestamp"));
        colMessage.setCellValueFactory(new PropertyValueFactory<>("message"));
        colSensor.setCellValueFactory(new PropertyValueFactory<>("stationName"));
        colStatus.setCellValueFactory(new PropertyValueFactory<>("status"));

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
        log.info("[STUB] Abriendo configuración de reglas...");
        new Alert(Alert.AlertType.INFORMATION, "Diálogo de reglas no implementado aún.").show();
    }

    @FXML
    public void onExportLog() {
        log.info("[STUB] Exportando log...");
    }
}