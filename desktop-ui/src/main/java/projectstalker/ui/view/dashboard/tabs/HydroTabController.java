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

import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.stream.Collectors; // Optional

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
    private ComboBox<SensorResponseDTO> virtualSensorSelector;

    @FXML
    private ComboBox<String> metricSelector;

    @FXML
    private Slider rainSlider;
    @FXML
    private Slider flowSlider;
    @FXML
    private LineChart<String, Number> levelChart;

    // Series para el gráfico
    private final XYChart.Series<String, Number> levelSeries = new XYChart.Series<>();

    private java.util.List<projectstalker.domain.dto.sensor.SensorReadingDTO> lastPollReadings = new java.util.ArrayList<>();

    @FXML
    public void initialize() {
        setupSourceSwitch();
        setupChart();
        setupListeners();
        setupVirtualSensorSelector();
        setupMetricSelector();

        // Auto-conectar al iniciar (Idealmente debería ser al abrir el proyecto)
        connectToRealTime();
    }

    private void setupMetricSelector() {
        if (metricSelector != null) {
            metricSelector.setOnAction(e -> {
                log.info("Métrica cambiada a: {}", metricSelector.getValue());
                updateChartFromLastReadings();
            });
        }
    }

    private void updateChartFromLastReadings() {
        if (lastPollReadings.isEmpty())
            return;

        Platform.runLater(() -> {
            String selectedMetric = metricSelector.getValue();

            // If nothing selected, pick the first available tag and select it
            if (selectedMetric == null || selectedMetric.isBlank()) {
                if (!lastPollReadings.isEmpty()) {
                    // Find most common or first tag
                    String firstTag = lastPollReadings.get(0).tag();
                    metricSelector.getSelectionModel().select(firstTag);
                    return; // The selection change will trigger this method again
                }
            }

            // Filter data
            final String metric = (selectedMetric == null) ? "value" : selectedMetric;

            java.util.List<projectstalker.domain.dto.sensor.SensorReadingDTO> filtered = lastPollReadings.stream()
                    .filter(r -> metric.equals(r.tag()) || "value".equals(metric) && "value".equals(r.tag()))
                    .sorted(java.util.Comparator
                            .comparing(projectstalker.domain.dto.sensor.SensorReadingDTO::timestamp))
                    .collect(Collectors.toList());

            boolean metricChanged = !metric.equals(levelSeries.getName());

            // Update Series Name/Axis
            if (metricChanged) {
                levelSeries.setName(metric);
                levelChart.getYAxis().setLabel("Valor (" + metric + ")");
            }

            if (filtered.isEmpty()) {
                if (metricChanged)
                    levelSeries.getData().clear();
                return;
            }

            // BULK vs INCREMENTAL Logic
            // If we have > 1 point (History/SAICA) OR we just changed metric (need to
            // re-render history)
            if (filtered.size() > 1 || metricChanged) {
                levelSeries.getData().clear();
                for (projectstalker.domain.dto.sensor.SensorReadingDTO r : filtered) {
                    String label = formatTimeLabel(r.timestamp());
                    levelSeries.getData().add(new XYChart.Data<>(label, r.value()));
                }
            } else {
                // INCREMENTAL (Standard Sensor returning 1 point typically)
                // Just append to existing history
                projectstalker.domain.dto.sensor.SensorReadingDTO r = filtered.get(0);
                String label = formatTimeLabel(r.timestamp());
                levelSeries.getData().add(new XYChart.Data<>(label, r.value()));

                // Keep limit
                if (levelSeries.getData().size() > 50)
                    levelSeries.getData().remove(0);
            }
        });
    }

    private String formatTimeLabel(String isoTimestamp) {
        try {
            LocalDateTime dt = LocalDateTime.parse(isoTimestamp);
            return dt.format(DateTimeFormatter.ofPattern("dd/MM HH:mm"));
        } catch (Exception e) {
            return isoTimestamp;
        }
    }

    private void updateMetricsList(java.util.List<projectstalker.domain.dto.sensor.SensorReadingDTO> readings) {
        if (readings.isEmpty())
            return;

        Platform.runLater(() -> {
            java.util.Set<String> tags = readings.stream()
                    .map(projectstalker.domain.dto.sensor.SensorReadingDTO::tag)
                    .collect(Collectors.toSet());

            // If new tags found, update combo
            // Careful not to reset selection if valid
            if (!tags.equals(new java.util.HashSet<>(metricSelector.getItems()))) {
                String selected = metricSelector.getValue();
                metricSelector.getItems().setAll(tags);
                if (selected != null && tags.contains(selected)) {
                    metricSelector.getSelectionModel().select(selected);
                } else if (!tags.isEmpty()) {
                    metricSelector.getSelectionModel().select(tags.iterator().next());
                }
            }
        });
    }

    private void setupVirtualSensorSelector() {
        if (virtualSensorSelector == null)
            return;

        virtualSensorSelector.setConverter(new StringConverter<SensorResponseDTO>() {
            @Override
            public String toString(SensorResponseDTO dto) {
                return dto == null ? null : dto.name() + " (Virtual)";
            }

            @Override
            public SensorResponseDTO fromString(String string) {
                return null;
            }
        });

        virtualSensorSelector.getSelectionModel().selectedItemProperty().addListener((obs, old, newVal) -> {
            if (newVal != null) {
                flowSensorSelector.getSelectionModel().clearSelection(); // Mutual rejection

                // Fetch Data from Backend (Simulation)
                sensorService.getHistory(newVal.stationId(), "NIVEL")
                        .subscribe(
                                fullData -> Platform.runLater(() -> plotSensorData(fullData)),
                                err -> log.error("Error fetching virtual history", err));
            }
        });

        // Update flow selector listener to clear virtual one
        flowSensorSelector.getSelectionModel().selectedItemProperty().addListener((obs, old, newVal) -> {
            if (newVal != null) {
                virtualSensorSelector.getSelectionModel().clearSelection();
                if (newVal.values() != null)
                    plotSensorData(newVal);
            }
        });
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

    private projectstalker.domain.dto.twin.TwinSummaryDTO currentTwin;

    public void setTwinContext(projectstalker.domain.dto.twin.TwinSummaryDTO twin) {
        this.currentTwin = twin;
        loadSensors();
    }

    private void loadSensors() {
        if (currentTwin == null)
            return;

        Platform.runLater(() -> {
            flowSensorSelector.getItems().clear();
            if (virtualSensorSelector != null)
                virtualSensorSelector.getItems().clear();
        });

        // Cargar Mocks (SAICA) + Reales (DB)
        var mocks = sensorService.getAllAvailableSensors();
        var reals = sensorService.getSensorsByTwin(currentTwin.id());

        // Merge flux
        reactor.core.publisher.Flux.merge(mocks, reals)
                .distinct(SensorResponseDTO::stationId) // Avoid duplicates if any
                .subscribe(sensor -> Platform.runLater(() -> {
                    // LOGGING DEBUG
                    log.info("Sensor DEBUG: Name='{}' Type='{}' Unit='{}'",
                            sensor.name(), sensor.signalType(), sensor.unit());

                    // Filter Logic: Virtual Level Sensors vs Flow/Others
                    // Check for "VIRTUAL" signalType OR "NIVEL" in name as fallback
                    boolean isVirtualLevel = "VIRTUAL".equalsIgnoreCase(sensor.signalType())
                            && ("m".equals(sensor.unit()) || sensor.name().toUpperCase().contains("NIVEL"));

                    if (isVirtualLevel && virtualSensorSelector != null) {
                        virtualSensorSelector.getItems().add(sensor);
                    } else {
                        flowSensorSelector.getItems().add(sensor);
                    }
                }), err -> log.error("Error cargando sensores híbridos", err));
    }

    @org.springframework.context.event.EventListener
    public void onSensorListRefresh(projectstalker.ui.event.SensorListRefreshEvent event) {
        log.info("Refrescando lista de sensores en Hidrodinámica...");
        loadSensors();
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

        // Listener logic moved to setupVirtualSensorSelector to handle mutual exclusion
    }

    private javafx.animation.Timeline activePoll;

    private void stopPolling() {
        if (activePoll != null) {
            activePoll.stop();
            activePoll = null;
        }
    }

    private void loadSensorData(String stationId) {
        // Stop any existing poll first
        stopPolling();

        // Clear previous context
        Platform.runLater(() -> {
            if (metricSelector != null)
                metricSelector.getItems().clear();
            lastPollReadings.clear();
        });

        log.info("Loading initial data for station: {}", stationId);

        // Initial Fetch
        sensorService.getRealtime(stationId, "ALL")
                .collectList()
                .subscribe(readings -> {
                    if (readings != null && !readings.isEmpty()) {
                        this.lastPollReadings = readings;
                        updateMetricsList(readings);
                        updateChartFromLastReadings();

                        // CHECK: Bulk History (SAICA) or Realtime Stream?
                        long distinctTimes = readings.stream()
                                .map(projectstalker.domain.dto.sensor.SensorReadingDTO::timestamp)
                                .distinct()
                                .count();

                        if (distinctTimes > 1) {
                            log.info("Station {} returned {} historical points. disabling polling.", stationId,
                                    distinctTimes);
                            // Do NOT start polling loop.
                        } else {
                            log.info("Station {} returned single point. Starting realtime polling.", stationId);
                            startPollingLoop(stationId);
                        }
                    } else {
                        log.warn("Hydro Poll: No readings found for {}. Retrying in loop...", stationId);
                        startPollingLoop(stationId);
                    }
                }, err -> log.error("Hydro initial load error: " + err.getMessage()));
    }

    private void startPollingLoop(String stationId) {
        stopPolling();
        activePoll = new javafx.animation.Timeline(
                new javafx.animation.KeyFrame(javafx.util.Duration.seconds(4), ev -> {
                    sensorService.getRealtime(stationId, "ALL")
                            .collectList()
                            .subscribe(readings -> {
                                if (readings != null && !readings.isEmpty()) {
                                    this.lastPollReadings = readings;
                                    updateMetricsList(readings);
                                    updateChartFromLastReadings();

                                    // Safety check: if we suddenly get bulk, stop?
                                    if (readings.size() > 10) {
                                        log.info("Switched to bulk data in poll. Stopping loop.");
                                        stopPolling();
                                    }
                                }
                            }, err -> {
                                log.error("Hydro poll error: " + err.getMessage());
                                // Backoff?
                            });
                }));
        activePoll.setCycleCount(javafx.animation.Animation.INDEFINITE);
        activePoll.play();
    }

    private void plotSensorData(SensorResponseDTO sensor) {
        levelChart.setTitle("Monitor Tiempo Real - " + sensor.name());
        levelChart.getYAxis().setLabel("Valor (" + sensor.unit() + ")");

        // Ensure series is attached
        if (!levelChart.getData().contains(levelSeries)) {
            levelChart.getData().add(levelSeries);
        }

        levelSeries.setName(sensor.name());
        levelSeries.getData().clear();

        // 1. Plot initial history if available
        if (sensor.values() != null) {
            for (projectstalker.domain.dto.sensor.SensorReadingDTO reading : sensor.values()) {
                String label = reading.timestamp().length() >= 16 ? reading.timestamp().substring(11, 16)
                        : reading.timestamp();
                levelSeries.getData().add(new XYChart.Data<>(label, reading.value()));
            }
        }

        // 2. Load Data (Smart Poll)
        loadSensorData(sensor.stationId());
    }

    private void updateVisibility(String mode) {
        boolean isManual = "MANUAL".equals(mode);
        manualControls.setVisible(isManual);
        manualControls.setManaged(isManual);
        sensorControls.setVisible(!isManual);
        sensorControls.setManaged(!isManual);
        log.info("Modo Hidrodinámica: {}", mode);
    }

    @org.springframework.context.event.EventListener
    public void onStationSelected(projectstalker.ui.event.StationSelectedEvent event) {
        String stationId = event.getStationId();
        log.info("Recibido evento de selección de estación: {}", stationId);

        Platform.runLater(() -> {
            // 1. Switch to "Real Time" mode if not already
            sourceModeGroup.getToggles().stream()
                    .filter(t -> "REAL".equals(t.getUserData()))
                    .findFirst()
                    .ifPresent(t -> sourceModeGroup.selectToggle(t));

            // 2. Select the sensor in the ComboBox (Check both lists)
            flowSensorSelector.getItems().stream()
                    .filter(dto -> dto.stationId().equals(stationId))
                    .findFirst()
                    .ifPresent(dto -> flowSensorSelector.getSelectionModel().select(dto));

            if (virtualSensorSelector != null) {
                virtualSensorSelector.getItems().stream()
                        .filter(dto -> dto.stationId().equals(stationId))
                        .findFirst()
                        .ifPresent(dto -> virtualSensorSelector.getSelectionModel().select(dto));
            }
        });
    }
}