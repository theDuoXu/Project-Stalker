package projectstalker.ui.view.dashboard;

import javafx.application.Platform;
import javafx.fxml.FXML;
import javafx.scene.control.Label;
import javafx.scene.control.Tab;
import javafx.scene.control.TabPane;
import javafx.scene.web.WebView;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.ApplicationEventPublisher;
import org.springframework.stereotype.Component;
import projectstalker.domain.dto.twin.TwinSummaryDTO;
import projectstalker.ui.event.RestoreMainViewEvent;
import projectstalker.ui.event.SidebarVisibilityEvent;
import projectstalker.ui.security.AuthenticationService;
import projectstalker.ui.view.dashboard.tabs.HydroTabController;
import projectstalker.ui.view.dashboard.tabs.QualityTabController;

import java.util.Set;

@Slf4j
@Component
@RequiredArgsConstructor
public class TwinDashboardController {

    private final AuthenticationService authService;
    private final ApplicationEventPublisher eventPublisher;
    // --- Header UI Components ---
    @FXML
    private Label twinNameLabel;
    @FXML
    private Label twinDescLabel;
    @FXML
    private Label hpcStatusLabel;

    // --- Tab System ---
    @FXML
    private TabPane mainTabPane;
    @FXML
    private Tab hydroTab; // Hidrodinámica
    @FXML
    private Tab qualityTab; // Calidad (Sensores)
    @FXML
    private Tab alertsTab; // Alertas
    @FXML
    private Tab reportsTab; // Informes
    @FXML
    private HydroTabController hydroViewController;
    @FXML
    private QualityTabController qualityViewController;
    private TwinSummaryDTO currentTwin;

    /**
     * Método de entrada principal. Llamado por MainController al abrir un río.
     */
    public void setTwin(TwinSummaryDTO twin) {
        this.currentTwin = twin;
        log.info("Inicializando Dashboard para Twin ID: {}", twin.id());

        // 1. Configuración Header
        Platform.runLater(() -> {
            this.twinNameLabel.setText(twin.name());
            this.twinDescLabel.setText(twin.description() != null ? twin.description() : "");
            updateHpcStatus(false);
        });

        // 2. PROPAGAR EL ID A LOS HIJOS (CRÍTICO)
        // Esto soluciona el error "Twin ID no establecido"
        if (qualityViewController != null) {
            qualityViewController.setTwinContext(twin);
        }

        if (hydroViewController != null) {
            hydroViewController.setTwinContext(twin);
        }

        // 3. Cargar Mapa y Seguridad
        applySecurityPolicies();
    }

    /**
     * Filtra qué pestañas ve el usuario basándose en sus roles.
     */
    private void applySecurityPolicies() {
        Set<String> roles = authService.getCurrentRoles();
        log.debug("Aplicando políticas de UI para roles: {}", roles);

        boolean isAdmin = roles.contains("ADMIN");
        boolean isAnalyst = roles.contains("ANALYST");
        boolean isTechnician = roles.contains("TECHNICIAN");
        boolean isOfficer = roles.contains("OFFICER");
        boolean isGuest = roles.contains("GUEST");

        Platform.runLater(() -> {
            // TAB 1: HIDRODINÁMICA -> Solo personal técnico
            if (!isAdmin && !isAnalyst && !isTechnician) {
                mainTabPane.getTabs().remove(hydroTab);
            }

            // TAB 3: ALERTAS -> Invitados fuera
            if (isGuest && !isAdmin && !isAnalyst && !isTechnician && !isOfficer) {
                mainTabPane.getTabs().remove(alertsTab);
            }

            // TAB 4: INFORMES -> Hidden for now (User Request)
            mainTabPane.getTabs().remove(reportsTab);

            // TAB 2 (CALIDAD/SENSORES) visible para todos (Guest ve solo lectura)
        });
    }

    @FXML
    public void reconnectHpc() {
        // TODO: Implementar lógica de reconexión WebSocket con el backend de
        // computación
        log.info("[STUB] Iniciando secuencia de reconexión manual con HPC Engine...");
    }

    @org.springframework.beans.factory.annotation.Autowired
    private org.springframework.context.ApplicationContext springContext;

    @FXML
    public void onCloseDashboard() {
        log.info("Cerrando Dashboard y volviendo al listado...");
        // Muestra la barra lateral de nuevo
        eventPublisher.publishEvent(new SidebarVisibilityEvent(true));
        // Limpia el área central (MainController escucha esto)
        eventPublisher.publishEvent(new RestoreMainViewEvent());
    }

    private void updateHpcStatus(boolean connected) {
        if (connected) {
            hpcStatusLabel.setText("HPC: CONECTADO");
            hpcStatusLabel.getStyleClass().removeAll("status-offline");
            hpcStatusLabel.getStyleClass().add("status-online");
            hpcStatusLabel.getStyleClass().add("status-offline");
        }
    }

    @org.springframework.context.event.EventListener
    public void onStationSelected(projectstalker.ui.event.StationSelectedEvent event) {
        Platform.runLater(() -> {
            if (mainTabPane != null && hydroTab != null) {
                // Si la pestaña actual no es Hydro, cambiar
                if (mainTabPane.getSelectionModel().getSelectedItem() != hydroTab) {
                    mainTabPane.getSelectionModel().select(hydroTab);
                }
            }
        });
    }
}