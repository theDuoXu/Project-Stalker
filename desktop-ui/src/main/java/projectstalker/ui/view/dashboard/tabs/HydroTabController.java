package projectstalker.ui.view.dashboard.tabs;

import javafx.fxml.FXML;
import javafx.scene.chart.LineChart;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Slider;
import javafx.scene.control.ToggleGroup;
import javafx.scene.layout.VBox;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

@Slf4j
@Component
public class HydroTabController {
    // Contenedores de los dos modos
    @FXML private VBox manualControls;
    @FXML private VBox sensorControls;

    // UI Elements
    @FXML private ToggleGroup sourceModeGroup;
    @FXML private ComboBox<String> flowSensorSelector;
    @FXML private Slider rainSlider;
    @FXML private Slider flowSlider;
    @FXML private LineChart<Number, Number> levelChart;

    @FXML
    public void initialize() {
        setupSourceSwitch();
    }

    private void setupChart() {
        // TODO: Configurar ejes, leyenda y estilos del gráfico en tiempo real
        levelChart.setAnimated(false); // Importante para rendimiento en tiempo real
    }

    private void setupListeners() {
        // TODO: Vincular sliders con el modelo de simulación local o enviar al HPC
        rainSlider.valueProperty().addListener((obs, oldVal, newVal) -> {
            log.debug("Slider Lluvia cambiado a: {}", newVal);
        });
    }

    @FXML
    public void onUpdateSimulation() {
        // TODO: Enviar parámetros actuales al Backend (HPC) vía WebSocket
        log.info("[STUB] Enviando parámetros de simulación: Lluvia={}, Caudal Base={}",
                rainSlider.getValue(), flowSlider.getValue());
    }

    private void setupSourceSwitch() {
        // Lógica de cambio de vista
        sourceModeGroup.selectedToggleProperty().addListener((obs, oldVal, newVal) -> {
            if (newVal == null) return;

            // Usamos userData en el FXML para saber qué modo es
            String mode = newVal.getUserData().toString();
            boolean isManual = "MANUAL".equals(mode);

            // Toggle de visibilidad y layout (managed)
            manualControls.setVisible(isManual);
            manualControls.setManaged(isManual);

            sensorControls.setVisible(!isManual);
            sensorControls.setManaged(!isManual);

            log.info("Modo de entrada Hidrodinámica cambiado a: {}", mode);
        });

        // Cargar dummy data en el selector de sensores reales
        flowSensorSelector.getItems().addAll("Sensor Cabecera (S-001)", "Aforo Estación Sur (S-042)");
    }
}