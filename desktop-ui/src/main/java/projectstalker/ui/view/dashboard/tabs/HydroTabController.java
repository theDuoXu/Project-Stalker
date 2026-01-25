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
import projectstalker.ui.view.dashboard.tabs.hydro.HydroChartManager;
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

    @FXML
    public void initialize() {
        // Init Managers
        this.chartManager = new HydroChartManager(levelChart);
        this.pollManager = new HydroPollManager(sensorService);

        setupSourceSwitch();
        // setupChart() removed, delegated to manager
        setupListeners();
        setupVirtualSensorSelector();
        setupMetricSelector();
        setupViewToggle();

        // Auto-conectar al iniciar
        connectToRealTime();

        // Cache: Preload all station data for instant map popups
        preloadData();
    }

    private void preloadData() {
        log.info("Iniciando precarga masiva de datos de estaciones (Conexión a Backend)...");

        // 1. Obtener lista de estaciones conocidas (Metadata)
        sensorService.getAllAvailableSensors()
                .map(SensorResponseDTO::stationId)
                .distinct()
                .flatMap(stationId ->
                // 2. Para cada estación, pedir datos REALES al backend
                sensorService.getRealtime(stationId, "ALL")
                        .collectList()
                        .map(readings -> new java.util.AbstractMap.SimpleEntry<>(stationId, readings))
                        // Si falla una, no detener todo
                        .onErrorResume(e -> {
                            log.warn("Fallo precarga para {}: {}", stationId, e.getMessage());
                            return reactor.core.publisher.Mono.just(
                                    new java.util.AbstractMap.SimpleEntry<>(stationId, new java.util.ArrayList<>()));
                        }), 5) // Concurrencia controlada
                .collectList()
                .subscribe(results -> {
                    log.info("Precarga: Recibidos datos de {} estaciones del backend.", results.size());
                    java.util.Map<String, String> popupCache = new java.util.HashMap<>();

                    for (java.util.Map.Entry<String, java.util.List<projectstalker.domain.dto.sensor.SensorReadingDTO>> entry : results) {
                        String stationId = entry.getKey();
                        java.util.List<projectstalker.domain.dto.sensor.SensorReadingDTO> readings = entry.getValue();

                        if (readings == null || readings.isEmpty()) {
                            continue;
                        }

                        // 3. Agregar todas las lecturas (último valor de cada tag)
                        java.util.Map<String, projectstalker.domain.dto.sensor.SensorReadingDTO> latestByTag = new java.util.HashMap<>();
                        for (projectstalker.domain.dto.sensor.SensorReadingDTO r : readings) {
                            latestByTag.merge(r.tag(), r, (oldVal, newVal) -> oldVal.timestamp()
                                    .compareTo(newVal.timestamp()) > 0 ? oldVal : newVal);
                        }

                        // 4. Generar HTML unificado
                        StringBuilder html = new StringBuilder();
                        html.append("<div style='min-width:150px'>");
                        html.append("<h4>").append(stationId).append("</h4>");
                        html.append("<table style='width:100%; font-size:12px; border-collapse: collapse;'>");

                        java.util.List<projectstalker.domain.dto.sensor.SensorReadingDTO> sorted = new java.util.ArrayList<>(
                                latestByTag.values());
                        sorted.sort(
                                java.util.Comparator.comparing(projectstalker.domain.dto.sensor.SensorReadingDTO::tag));

                        for (projectstalker.domain.dto.sensor.SensorReadingDTO r : sorted) {
                            html.append("<tr>");
                            html.append("<td style='color:#aad;'>").append(r.tag()).append(":</td>");
                            html.append("<td style='text-align:right; font-weight:bold;'>")
                                    .append(String.format("%.2f", r.value())).append("</td>");
                            html.append("</tr>");
                        }
                        html.append("</table>");
                        // Footer time from first reading
                        if (!sorted.isEmpty()) {
                            String ts = sorted.get(0).timestamp().replace("T", " ");
                            if (ts.length() > 16)
                                ts = ts.substring(5, 16); // mm-dd HH:mm
                            html.append("<div style='font-size:10px; color:#888; margin-top:5px; text-align:right;'>")
                                    .append(ts)
                                    .append("</div>");
                        }
                        html.append("</div>");

                        popupCache.put(stationId, html.toString());
                    }

                    log.info("Precarga finalizada. Generado cache para {} estaciones. Inyectando...",
                            popupCache.size());
                    Platform.runLater(() -> {
                        if (embeddedMapController != null) {
                            embeddedMapController.injectBulkData(popupCache);
                        }
                    });
                }, err -> log.error("Error crítico en precarga masiva", err));
    }

    private void setupViewToggle() {
        if (viewToggle != null) {
            viewToggle.selectedProperty().addListener((obs, oldVal, newVal) -> {
                boolean showMap = newVal;

                // Toggle Visibility
                if (levelChart != null) {
                    levelChart.setVisible(!showMap);
                    levelChart.setManaged(!showMap);
                }

                if (embeddedMap != null) {
                    embeddedMap.setVisible(showMap);
                    embeddedMap.setManaged(showMap);
                    // Note: We do NOT call loadData() here anymore to preserve the cache.
                }

                viewToggle.setText(showMap ? "Ver Gráfica de Datos" : "Ver Mapa Geoespacial");
                org.kordamp.ikonli.javafx.FontIcon icon = (org.kordamp.ikonli.javafx.FontIcon) viewToggle.getGraphic();
                if (icon != null)
                    icon.setIconLiteral(showMap ? "mdi2c-chart-line" : "mdi2m-map");
            });
        }
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
        Platform.runLater(() -> {
            if (lastPollReadings.isEmpty())
                return;

            String selectedMetric = metricSelector.getValue();

            // Auto-select first tag if nothing selected
            if (selectedMetric == null || selectedMetric.isBlank()) {
                String firstTag = lastPollReadings.get(0).tag();
                metricSelector.getSelectionModel().select(firstTag);
                return;
            }

            // Delegate to Manager
            chartManager.updateData(lastPollReadings, selectedMetric);
        });
    }

    // private String formatTimeLabel(String isoTimestamp) { // Removed
    // try {
    // LocalDateTime dt = LocalDateTime.parse(isoTimestamp);
    // return dt.format(DateTimeFormatter.ofPattern("dd/MM HH:mm"));
    // } catch (Exception e) {
    // return isoTimestamp;
    // }
    // }

    private void updateMetricsList(java.util.List<projectstalker.domain.dto.sensor.SensorReadingDTO> readings,
            String preferredMetric) {
        if (readings.isEmpty())
            return;

        Platform.runLater(() -> {
            java.util.Set<String> tags = readings.stream()
                    .map(projectstalker.domain.dto.sensor.SensorReadingDTO::tag)
                    .collect(Collectors.toSet());

            if (!tags.equals(new java.util.HashSet<>(metricSelector.getItems()))) {
                metricSelector.getItems().setAll(tags);

                // Try to restore preference
                if (preferredMetric != null && tags.contains(preferredMetric)) {
                    metricSelector.getSelectionModel().select(preferredMetric);
                } else if (!tags.isEmpty()) {
                    metricSelector.getSelectionModel().select(tags.iterator().next());
                }
            } else {
                // Even if items didn't change, ensure selection is valid if we had a preference
                String current = metricSelector.getValue();
                if ((current == null || current.isBlank()) && preferredMetric != null
                        && tags.contains(preferredMetric)) {
                    metricSelector.getSelectionModel().select(preferredMetric);
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
                if (embeddedMap != null && embeddedMap.isVisible()) {
                    embeddedMapController.focusStation(newVal.stationId());
                }
                // Do not auto-plot here, setupVirtualSensorSelector handles it or mutual
                // exclusion logic
                // Actually setupVirtualSensorSelector logic handles plotting
            }
        });
    }

    // private javafx.animation.Timeline activePoll; // Removed

    // private void stopPolling() { // Removed
    // if (activePoll != null) {
    // activePoll.stop();
    // activePoll = null;
    // }
    // }

    // --- Data Loading Logic ---

    private void loadSensorData(String stationId) {
        pollManager.stopPolling();

        final String preferredMetric = (metricSelector != null) ? metricSelector.getValue() : null;

        Platform.runLater(() -> {
            if (metricSelector != null)
                metricSelector.getItems().clear();
            lastPollReadings.clear();
            chartManager.clear();
        });

        log.info("Loading initial data for station: {}", stationId);

        // Initial Fetch using Service directly (Manager could do this too, but we need
        // readings here to update metrics UI)
        sensorService.getRealtime(stationId, "ALL")
                .collectList()
                .subscribe(readings -> {
                    if (readings != null && !readings.isEmpty()) {
                        this.lastPollReadings = readings;
                        updateMetricsList(readings, preferredMetric);
                        updateChartFromLastReadings();

                        long distinctTimes = readings.stream()
                                .map(projectstalker.domain.dto.sensor.SensorReadingDTO::timestamp)
                                .distinct().count();

                        if (distinctTimes > 1) {
                            log.info("Station {} is historical. disabling polling.", stationId);
                        } else {
                            log.info("Station {} is realtime. polling started.", stationId);
                            pollManager.startPolling(stationId, this::onPollDataReceived);
                        }
                    } else {
                        log.warn("No initial readings. Starting poll anyway...", stationId);
                        pollManager.startPolling(stationId, this::onPollDataReceived);
                    }
                }, err -> log.error("Hydro initial load error: " + err.getMessage()));
    }

    private void onPollDataReceived(java.util.List<projectstalker.domain.dto.sensor.SensorReadingDTO> readings) {
        this.lastPollReadings = readings;
        updateMetricsList(readings, metricSelector.getValue()); // Keep current if possible
        updateChartFromLastReadings();

        // Update Map Popup if Map is visible
        if (embeddedMap != null && embeddedMap.isVisible() && embeddedMapController != null && !readings.isEmpty()) {
            StringBuilder html = new StringBuilder();
            html.append("<div style='min-width:150px'>");
            // Header (Station Name/ID from first reading)
            html.append("<h4>").append(readings.get(0).stationId()).append("</h4>");
            html.append("<table style='width:100%; font-size:12px; border-collapse: collapse;'>");

            // Group by Tag to show latest value per metric
            java.util.Map<String, projectstalker.domain.dto.sensor.SensorReadingDTO> latestByTag = readings.stream()
                    .collect(Collectors.toMap(
                            projectstalker.domain.dto.sensor.SensorReadingDTO::tag,
                            r -> r,
                            (existing, replacement) -> existing.timestamp().compareTo(replacement.timestamp()) > 0
                                    ? existing
                                    : replacement));

            // Sort by tag for consistent display
            java.util.List<projectstalker.domain.dto.sensor.SensorReadingDTO> sortedList = new java.util.ArrayList<>(
                    latestByTag.values());
            sortedList.sort(java.util.Comparator.comparing(projectstalker.domain.dto.sensor.SensorReadingDTO::tag));

            for (projectstalker.domain.dto.sensor.SensorReadingDTO r : sortedList) {
                html.append("<tr>");
                html.append("<td style='color:#aad;'>").append(r.tag()).append(":</td>");
                html.append("<td style='text-align:right; font-weight:bold;'>")
                        .append(String.format("%.2f", r.value())).append("</td>");
                html.append("</tr>");
            }
            html.append("</table>");
            html.append("<div style='font-size:10px; color:#888; margin-top:5px; text-align:right;'>")
                    .append(readings.get(0).timestamp().replace("T", " ")) // Simple formatting
                    .append("</div>");
            html.append("</div>");

            embeddedMapController.updatePopup(readings.get(0).stationId(), html.toString());
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
        // Make sure series is managed
        chartManager.setMetricName(sensor.unit()); // Fallback name

        // Use Manager to clear/prep
        chartManager.clear();

        // 1. Initial history from DTO
        if (sensor.values() != null) {
            chartManager.updateData(
                    sensor.values().stream().map(v -> new projectstalker.domain.dto.sensor.SensorReadingDTO(
                            "value", v.timestamp(), v.value(), String.valueOf(v.value()), sensor.stationId()))
                            .collect(Collectors.toList()),
                    "value");
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
                    // Already selected, force reload
                    plotSensorData(flowMatch.get());
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