package projectstalker.ui.view.dashboard.tabs;

import javafx.application.Platform;
import javafx.fxml.FXML;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.XYChart;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Slider;
import javafx.scene.control.ToggleGroup;
import javafx.scene.layout.VBox;
import javafx.util.StringConverter;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import projectstalker.domain.dto.sensor.SensorResponseDTO;
import projectstalker.domain.simulation.SimulationResponseDTO;
import projectstalker.ui.service.RealTimeClientService;
import projectstalker.ui.service.SensorClientService;

import java.time.LocalTime;
import java.time.format.DateTimeFormatter;

@Slf4j
@Component
@RequiredArgsConstructor
public class HydroTabController {

    private final RealTimeClientService realTimeService;
    private final SensorClientService sensorService;

    // Contenedores de los dos modos
    @FXML
    private VBox manualControls;
    @FXML
    private VBox sensorControls;

    // UI Elements
    @FXML
    private ToggleGroup sourceModeGroup;
    @FXML
    private ComboBox<SensorResponseDTO> flowSensorSelector;
    @FXML
    private Slider rainSlider;
    @FXML
    private Slider flowSlider;
    @FXML
    private LineChart<String, Number> levelChart;

    // Series para el gráfico
    private final XYChart.Series<String, Number> levelSeries = new XYChart.Series<>();

    @FXML
    public void initialize() {
        setupSourceSwitch();
        setupChart();
        setupListeners();

        // Auto-conectar al iniciar (Idealmente debería ser al abrir el proyecto)
        connectToRealTime();
    }

    private void connectToRealTime() {
        // Conectar al socket
        realTimeService.connect();

        // Suscribirse a una simulación de demo (ID hardcoded para probar)
        String simId = "sim-demo-001";
        log.info("Suscribiéndose a simulación: {}", simId);

        realTimeService.subscribeToSimulation(simId)
                .subscribe(this::updateChart);
    }

    private void updateChart(SimulationResponseDTO dto) {
        Platform.runLater(() -> {
            String timeLabel = LocalTime.now().format(DateTimeFormatter.ofPattern("HH:mm:ss"));
            // Asumimos que el DTO trae algún valor representativo, por ejemplo el nivel
            // medio o el último paso
            // Como SimulationResponseDTO es un resumen, quizás necesitamos un DTO más
            // detallado para streaming (SimulationStepDTO).
            // Por ahora usaremos 'timestep' como valor dummy si no tiene datos reales de
            // nivel.

            // FIXME: SimulationResponseDTO no tiene datos de paso actual.
            // Como la simulación está fuera de alcance para esta release, generamos un
            // valor visual dummy.
            // Usamos una onda sinusoidal basada en el tiempo para que parezca "vivo".
            double mockValue = 2.0 + Math.sin(System.currentTimeMillis() / 1000.0) * 0.5 + (Math.random() * 0.1);
            Number value = mockValue;

            levelSeries.getData().add(new XYChart.Data<>(timeLabel, value));

            // Limitar a los últimos 20 puntos para que no explote la memoria
            if (levelSeries.getData().size() > 20) {
                levelSeries.getData().remove(0);
            }
        });
    }

    private void setupChart() {
        levelChart.setAnimated(false); // Importante para rendimiento en tiempo real
        levelChart.setTitle("Nivel de Agua (Tiempo Real)");
        levelSeries.setName("Profundidad (m)");
        levelChart.getData().add(levelSeries);
    }

    private void setupListeners() {
        rainSlider.valueProperty().addListener((obs, oldVal, newVal) -> {
            // Debounce o log
        });
    }

    @FXML
    public void onUpdateSimulation() {
        log.info("Enviando nuevos parámetros: Lluvia={}, Caudal={} (Pendiente de impl. envio)",
                rainSlider.getValue(), flowSlider.getValue());
        // Aquí llamaríamos a realTimeService.sendParameters(...)
    }

    private void setupSourceSwitch() {
        if (sourceModeGroup.getSelectedToggle() != null) {
            updateVisibility(sourceModeGroup.getSelectedToggle().getUserData().toString());
        }

        sourceModeGroup.selectedToggleProperty().addListener((obs, oldVal, newVal) -> {
            if (newVal != null) {
                updateVisibility(newVal.getUserData().toString());
            }
        });

        // Configurar selector de sensores
        flowSensorSelector.setConverter(new StringConverter<SensorResponseDTO>() {
            @Override
            public String toString(SensorResponseDTO dto) {
                return dto == null ? null : dto.name() + " (" + dto.stationId() + ")";
            }

            @Override
            public SensorResponseDTO fromString(String string) {
                return null; // No editable
            }
        });

        // Cargar datos de sensores disponibles (desde JSON local simulado en servicio)
        sensorService.getAllAvailableSensors()
                .subscribe(sensor -> Platform.runLater(() -> flowSensorSelector.getItems().add(sensor)));
    }

    private void updateVisibility(String mode) {
        boolean isManual = "MANUAL".equals(mode);
        manualControls.setVisible(isManual);
        manualControls.setManaged(isManual);
        sensorControls.setVisible(!isManual);
        sensorControls.setManaged(!isManual);
        log.info("Modo Hidrodinámica: {}", mode);
    }
}