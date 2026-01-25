package projectstalker.ui.view.dashboard.tabs;

import javafx.application.Platform;
import javafx.fxml.FXML;
import javafx.scene.chart.LineChart;

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
import projectstalker.ui.view.dashboard.tabs.hydro.HydroChartManager;
import projectstalker.ui.view.dashboard.tabs.hydro.HydroHistoryManager;
import projectstalker.ui.view.dashboard.tabs.hydro.HydroMapManager;
import projectstalker.ui.view.dashboard.tabs.hydro.HydroPollManager;

import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.stream.Collectors;

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

    private java.util.List<projectstalker.domain.dto.sensor.SensorReadingDTO> lastPollReadings = new java.util.ArrayList<>();

    @FXML
    private javafx.scene.control.ToggleButton viewToggle;
    @FXML
    private javafx.scene.layout.AnchorPane embeddedMap;
    // Injected via fx:include + Controller naming convention
    @FXML
    private LeafletMapController embeddedMapController;

    private HydroChartManager chartManager;
    private HydroPollManager pollManager;

    // Navigation UI
    @FXML
    private javafx.scene.control.Button prevWindowBtn;
    @FXML
    private javafx.scene.control.Button nextWindowBtn;
    // Managers
    // chartManager & pollManager are defined above (lines 70-71)
    private HydroMapManager mapManager;
    private HydroHistoryManager historyManager;

    @FXML
    public void initialize() {
        // Init Managers
        this.chartManager = new HydroChartManager(levelChart);
        this.pollManager = new HydroPollManager(sensorService);
        this.mapManager = new HydroMapManager(sensorService, embeddedMapController);
        this.historyManager = new HydroHistoryManager(sensorService);

        setupSourceSwitch();
        setupListeners();
        setupVirtualSensorSelector();
        setupMetricSelector();

        // Delegate to Managers
        this.mapManager.setupViewToggle(viewToggle, levelChart.getParent(), levelChart.getParent(), embeddedMap);

        // Auto-conectar al iniciar
        connectToRealTime();

        // Cache: Preload all station data for instant map popups
        this.mapManager.preloadData();
    }

    private void refreshCurrentData() {
        // Reduced logic: Just load current selected station data
        String currentStation = getSelectedStationId();
        if (currentStation == null)
            return;

        loadSensorData(currentStation);
    }

    private String getSelectedStationId() {
        // Try flow selector first, then virtual
        if (flowSensorSelector.getValue() != null)
            return flowSensorSelector.getValue().stationId();
        if (virtualSensorSelector != null && virtualSensorSelector.getValue() != null)
            return virtualSensorSelector.getValue().stationId();
        return null;
    }

    private String getSelectedStationName() {
        if (flowSensorSelector.getValue() != null)
            return flowSensorSelector.getValue().name();
        if (virtualSensorSelector != null && virtualSensorSelector.getValue() != null)
            return virtualSensorSelector.getValue().name();
        return "Desconocida";
    }

    private void updateChartTitle() {
        String stationName = getSelectedStationName();
        String metric = metricSelector.getValue();
        if (metric == null || metric.isBlank())
            metric = "General";

        levelChart.setTitle(String.format("%s - %s", stationName, metric));
    }

    private boolean isUpdatingMetrics = false;

    private void setupMetricSelector() {
        if (metricSelector != null) {
            metricSelector.setOnAction(e -> {
                if (isUpdatingMetrics)
                    return; // Prevent loop
                log.info("Métrica cambiada a: {}", metricSelector.getValue());
                refreshCurrentData(); // Reload with new metric logic or history re-fetch
            });
        }
    }

    private void updateChartFromLastReadings() {
        Platform.runLater(() -> {
            updateChartTitle(); // Update Title

            if (lastPollReadings.isEmpty())
                return;

            String selectedMetric = metricSelector.getValue();

            // Auto-select first tag if nothing selected
            if (selectedMetric == null || selectedMetric.isBlank()) {
                String firstTag = lastPollReadings.get(0).tag();
                // Avoid triggering listener if we just set default
                // But here we might WANT to trigger it to load data for that metric?
                // Actually loadSensorData fetches ALL, so chart update is enough.
                // Just update selection without firing?
                isUpdatingMetrics = true;
                try {
                    metricSelector.getSelectionModel().select(firstTag);
                } finally {
                    isUpdatingMetrics = false;
                }
                return;
            }

            // Delegate to Manager (Realtime: respect tags)
            chartManager.updateData(lastPollReadings, selectedMetric, false);
        });
    }

    private void updateMetricsList(java.util.List<projectstalker.domain.dto.sensor.SensorReadingDTO> readings,
            String preferredMetric) {
        if (readings.isEmpty())
            return;

        Platform.runLater(() -> {
            java.util.Set<String> tags = readings.stream()
                    .map(projectstalker.domain.dto.sensor.SensorReadingDTO::tag)
                    .collect(Collectors.toSet());

            if (!tags.equals(new java.util.HashSet<>(metricSelector.getItems()))) {
                log.info("Updating metrics list. New tags: {}", tags);
                isUpdatingMetrics = true;
                try {
                    metricSelector.getItems().setAll(tags);

                    // Try to restore preference
                    if (preferredMetric != null && tags.contains(preferredMetric)) {
                        metricSelector.getSelectionModel().select(preferredMetric);
                    } else if (!tags.isEmpty()) {
                        metricSelector.getSelectionModel().select(tags.iterator().next());
                    }
                } finally {
                    isUpdatingMetrics = false;
                }
            } else {
                // Even if items didn't change, ensure selection is valid if we had a preference
                String current = metricSelector.getValue();
                if ((current == null || current.isBlank()) && preferredMetric != null
                        && tags.contains(preferredMetric)) {
                    isUpdatingMetrics = true;
                    try {
                        metricSelector.getSelectionModel().select(preferredMetric);
                    } finally {
                        isUpdatingMetrics = false;
                    }
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

                // Fetch Data from Backend (Simulation) via History Manager or direct if it's
                // considered "History" browse action?
                // Actually Virtual sensors usually have history too.
                // For simplicity, treat as selection change -> plotSensorData
                plotSensorData(newVal);
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
        // RealTime Simulation Stream
        Platform.runLater(() -> {
            String timeLabel = LocalTime.now().format(DateTimeFormatter.ofPattern("HH:mm:ss"));
            // Mock value for demo
            double mockValue = 2.0 + Math.sin(System.currentTimeMillis() / 1000.0) * 0.5 + (Math.random() * 0.1);

            chartManager.addPoint(timeLabel, mockValue);
        });
    }

    // private void setupChart() { // Removed
    // levelChart.setAnimated(false); // Importante para rendimiento en tiempo real
    // levelChart.setTitle("Nivel de Agua (Tiempo Real)");
    // levelSeries.setName("Profundidad (m)");
    // levelChart.getData().add(levelSeries);
    // }

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

        // Listener for Sidebar Selection -> Map Focus
        flowSensorSelector.getSelectionModel().selectedItemProperty().addListener((obs, old, newVal) -> {
            if (newVal != null) {
                log.info("Station selected: {}", newVal.stationId());
                mapManager.focusStation(newVal.stationId());
                loadSensorData(newVal.stationId()); // Trigger data load
            }
        });
    }

    // private javafx.animation.Timeline activePoll; // Removed

    // private void stopPolling() { // Removed
    // if (activePoll != null) {
    // activePoll = null;
    // }
    // }

    // --- Data Loading Logic ---

    private void loadSensorData(String stationId) {
        pollManager.stopPolling();

        final String preferredMetric = (metricSelector != null) ? metricSelector.getValue() : null;

        Platform.runLater(() -> {
            if (metricSelector != null) {
                isUpdatingMetrics = true;
                try {
                    metricSelector.getItems().clear();
                } finally {
                    isUpdatingMetrics = false;
                }
            }
            lastPollReadings = new java.util.ArrayList<>();
            chartManager.clear();
        });

        log.info("Loading initial data for station: {}", stationId);

        // Initial Fetch using Service directly (Manager could do this too, but we need
        // readings here to update metrics UI)
        sensorService.getRealtime(stationId, "ALL")
                .collectList()
                .subscribe(readings -> {
                    log.info("DEBUG: getRealtime returned {} readings for {}",
                            readings == null ? "null" : readings.size(), stationId);
                    if (readings != null && !readings.isEmpty()) {
                        // Filter out invalid readings
                        java.util.List<projectstalker.domain.dto.sensor.SensorReadingDTO> validReadings = readings
                                .stream()
                                .filter(r -> r != null && r.timestamp() != null && !r.timestamp().isBlank())
                                .collect(Collectors.toList());

                        this.lastPollReadings = validReadings;
                        updateMetricsList(validReadings, preferredMetric);
                        updateChartFromLastReadings();

                        long distinctTimes = validReadings.stream()
                                .map(projectstalker.domain.dto.sensor.SensorReadingDTO::timestamp)
                                .distinct().count();

                        if (distinctTimes > 1) {
                            log.info("Station {} is historical. disabling polling.", stationId);
                        } else {
                            log.info("Station {} is realtime. polling started.", stationId);
                            pollManager.startPolling(stationId, this::onPollDataReceived);
                        }
                    } else {
                        log.warn("No initial readings for {}. Metrics list will be empty.", stationId);
                        pollManager.startPolling(stationId, this::onPollDataReceived);
                    }
                }, err -> log.error("Hydro initial load error: " + err.getMessage(), err));
    }

    private void onPollDataReceived(java.util.List<projectstalker.domain.dto.sensor.SensorReadingDTO> readings) {
        // Filter invalid readings
        java.util.List<projectstalker.domain.dto.sensor.SensorReadingDTO> validReadings = (readings == null)
                ? java.util.Collections.emptyList()
                : readings.stream()
                        .filter(r -> r != null && r.timestamp() != null && !r.timestamp().isBlank())
                        .collect(Collectors.toList());

        log.debug("onPollDataReceived: {} valid readings", validReadings.size());
        this.lastPollReadings = validReadings;
        updateMetricsList(validReadings, metricSelector.getValue()); // Keep current if possible
        updateChartFromLastReadings();

        // Update Map Popup if Map is visible (Delegated to Manager)
        if (readings != null && !readings.isEmpty()) {
            mapManager.updatePopup(readings.get(0).stationId(), readings);
        }
    }

    // private void startPollingLoop(String stationId) { // Removed
    // stopPolling();
    // activePoll = new javafx.animation.Timeline(
    // new javafx.animation.KeyFrame(javafx.util.Duration.seconds(4), ev -> {
    // sensorService.getRealtime(stationId, "ALL")
    // .collectList()
    // .subscribe(readings -> {
    // if (readings != null && !readings.isEmpty()) {
    // this.lastPollReadings = readings;
    // updateMetricsList(readings);
    // updateChartFromLastReadings();

    // // Safety check: if we suddenly get bulk, stop?
    // if (readings.size() > 10) {
    // log.info("Switched to bulk data in poll. Stopping loop.");
    // stopPolling();
    // }
    // }
    // }, err -> {
    // log.error("Hydro poll error: " + err.getMessage());
    // // Backoff?
    // });
    // }));
    // activePoll.setCycleCount(javafx.animation.Animation.INDEFINITE);
    // activePoll.play();
    // }

    private void plotSensorData(SensorResponseDTO sensor) {
        // Reset window to NOW when selecting new sensor

        // Make sure series is managed
        chartManager.setMetricName(sensor.unit()); // Fallback name

        // Use Manager to clear/prep
        chartManager.clear();

        // 1. Initial history from DTO
        if (sensor.values() != null) {
            chartManager.updateData(
                    sensor.values().stream()
                            .filter(v -> v != null && v.timestamp() != null && !v.timestamp().isBlank())
                            .map(v -> new projectstalker.domain.dto.sensor.SensorReadingDTO(
                                    "value", v.timestamp(), v.value(), String.valueOf(v.value()), sensor.stationId()))
                            .collect(Collectors.toList()),
                    "value", true);
        }

        // 2. Load Real
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
            java.util.Optional<SensorResponseDTO> flowMatch = flowSensorSelector.getItems().stream()
                    .filter(dto -> dto.stationId().equals(stationId))
                    .findFirst();

            if (flowMatch.isPresent()) {
                if (flowMatch.get().equals(flowSensorSelector.getValue())) {
                    // Already selected, force reload manually because listener won't fire
                    log.info("Station {} already selected in dropdown. Forcing reload.", stationId);
                    mapManager.focusStation(stationId);
                    loadSensorData(stationId);
                } else {
                    flowSensorSelector.getSelectionModel().select(flowMatch.get());
                }
            }

            if (virtualSensorSelector != null) {
                java.util.Optional<SensorResponseDTO> virtMatch = virtualSensorSelector.getItems().stream()
                        .filter(dto -> dto.stationId().equals(stationId))
                        .findFirst();

                if (virtMatch.isPresent()) {
                    if (virtMatch.get().equals(virtualSensorSelector.getValue())) {
                        // Mutual exclusion handled by listener usually, but here we force
                        flowSensorSelector.getSelectionModel().clearSelection();
                        plotSensorData(virtMatch.get());
                    } else {
                        virtualSensorSelector.getSelectionModel().select(virtMatch.get());
                    }
                }
            }
        });
    }
}