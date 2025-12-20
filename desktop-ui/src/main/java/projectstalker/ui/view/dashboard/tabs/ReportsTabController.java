package projectstalker.ui.view.dashboard.tabs;

import javafx.collections.FXCollections;
import javafx.fxml.FXML;
import javafx.scene.control.ComboBox;
import javafx.scene.control.DatePicker;
import javafx.scene.control.ListView;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.time.LocalDate;

@Slf4j
@Component
public class ReportsTabController {

    @FXML private DatePicker startDatePicker;
    @FXML private DatePicker endDatePicker;
    @FXML private ComboBox<String> reportTypeCombo;
    @FXML private ListView<String> generatedReportsList;

    @FXML
    public void initialize() {
        setupForm();
        loadHistory();
    }

    private void setupForm() {
        reportTypeCombo.setItems(FXCollections.observableArrayList(
                "Resumen Ejecutivo (PDF)",
                "Datos Crudos Hidrodinámica (CSV)",
                "Calidad de Agua - Detallado (PDF)",
                "Log de Auditoría (CSV)"
        ));

        // Default values
        startDatePicker.setValue(LocalDate.now().minusDays(7));
        endDatePicker.setValue(LocalDate.now());
    }

    private void loadHistory() {
        // TODO: Cargar lista de informes generados previamente
        generatedReportsList.getItems().add("Informe_Mensual_Noviembre.pdf (Generado: 01/12/2025)");
    }

    @FXML
    public void onGeneratePdf() {
        // TODO: Lanzar tarea asíncrona de generación de informe en Backend
        log.info("[STUB] Solicitando generación de PDF. Tipo: {}, Rango: {} - {}",
                reportTypeCombo.getValue(), startDatePicker.getValue(), endDatePicker.getValue());
    }

    @FXML
    public void onExportCsv() {
        // TODO: Descargar CSV directo
        log.info("[STUB] Solicitando exportación CSV...");
    }
}