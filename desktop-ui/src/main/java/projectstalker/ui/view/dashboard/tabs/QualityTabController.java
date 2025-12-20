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

@Slf4j
@Component
@RequiredArgsConstructor
public class QualityTabController {

    private final AuthenticationService authService;
    private final ApplicationContext springContext; // Necesario para inyectar el Wizard

    @FXML private StackPane heatmapContainer;
    @FXML private FlowPane sensorsFlowPane;
    @FXML private Button addSensorBtn;

    // IMPORTANTE: Necesitamos saber en qué río estamos
    @Setter
    private TwinSummaryDTO currentTwinContext;

    @FXML
    public void initialize() {
        checkPermissions();
        loadHeatmapPlaceholder();
        // loadSensorsList(); // Lo llamaremos cuando setTwinId sea invocado desde el padre
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
            });

            Stage stage = new Stage();
            stage.initModality(Modality.APPLICATION_MODAL);
            stage.setTitle("Nuevo Sensor - " + currentTwinContext.name());
            stage.setScene(new Scene(root));
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
        // TODO: Llamar a sensorService.getAllByTwin(currentTwinId)
        // Por ahora limpiamos para probar el flujo de creación visual
        sensorsFlowPane.getChildren().clear();
        log.info("[STUB] Lista de sensores recargada para Twin: {}", currentTwinContext.id());
    }

    private void loadHeatmapPlaceholder() {
        // STUB
    }
}