package projectstalker.ui.view.dashboard.tabs;

import javafx.fxml.FXML;
import javafx.scene.control.TableView;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

@Slf4j
@Component
public class AlertsTabController {

    @FXML private TableView<?> alertsTable; // Usar DTO específico en el futuro (AlertDTO)

    @FXML
    public void initialize() {
        // TODO: Configurar columnas del TableView (CellValueFactories)
        loadAlerts();
    }

    private void loadAlerts() {
        // TODO: Obtener histórico de alertas desde API REST
        log.info("[STUB] Cargando histórico de incidencias...");
    }

    @FXML
    public void onConfigureRules() {
        // TODO: Abrir diálogo de configuración de umbrales
        log.info("[STUB] Abriendo configuración de reglas de alerta...");
    }

    @FXML
    public void onExportLog() {
        // TODO: Generar CSV/Excel
        log.info("[STUB] Exportando log de alertas...");
    }
}