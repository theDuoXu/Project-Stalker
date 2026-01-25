package projectstalker.ui.view.dashboard.tabs;

import javafx.concurrent.Worker;
import javafx.application.Platform;
import javafx.fxml.FXML;
import javafx.scene.web.WebEngine;
import javafx.scene.web.WebView;
import lombok.extern.slf4j.Slf4j;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Component;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.ApplicationEventPublisher;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.stream.Collectors;

@Slf4j
@Component
@org.springframework.context.annotation.Scope("prototype")
public class LeafletMapController {

    @Autowired
    private ApplicationEventPublisher eventPublisher;

    @FXML
    private WebView mapWebView;

    private boolean mapReady = false;
    private java.util.Map<String, String> pendingBulkData;

    @FXML
    public void initialize() {
        log.info("Inicializando LeafletMapController...");
        initWebView();
    }

    private void initWebView() {
        WebEngine engine = mapWebView.getEngine();
        engine.setJavaScriptEnabled(true);

        // Cargar el HTML local
        String htmlPath = getClass().getResource("/html/map.html").toExternalForm();
        engine.load(htmlPath);

        // Ocultar WebView hasta que cargue para evitar pantallazo blanco
        mapWebView.setOpacity(0);
        mapWebView.setPageFill(javafx.scene.paint.Color.TRANSPARENT);

        // Listener de carga
        engine.getLoadWorker().stateProperty().addListener((obs, oldState, newState) -> {
            if (newState == Worker.State.SUCCEEDED) {
                log.info("Mapa cargado correctamente en WebView.");

                // Inject Java Bridge (Interaction)
                netscape.javascript.JSObject window = (netscape.javascript.JSObject) engine.executeScript("window");
                window.setMember("javaConnector", new JavaBridge(eventPublisher));

                mapReady = true;
                loadData();

                // Inject Pending Bulk Data if any
                if (pendingBulkData != null) {
                    log.info("Injecting pending bulk data ({} stations)", pendingBulkData.size());
                    injectBulkData(pendingBulkData);
                    pendingBulkData = null;
                }

                // Fade In effect
                javafx.animation.FadeTransition fade = new javafx.animation.FadeTransition(
                        javafx.util.Duration.millis(500), mapWebView);
                fade.setFromValue(0);
                fade.setToValue(1);
                fade.play();

                // First Render Fix
                refreshMap();

            } else if (newState == Worker.State.FAILED) {
                log.error("Fallo al cargar el mapa HTML.");
            }
        });

        // Visibility Listener: When toggled ON, force redraw
        mapWebView.visibleProperty().addListener((obs, oldVal, newVal) -> {
            if (newVal && mapReady) {
                refreshMap();
            }
        });

        // Resize Listener: Force map update when window/webview resizing
        javafx.beans.value.ChangeListener<Number> resizeListener = (obs, oldVal, newVal) -> {
            if (mapReady) {
                refreshMap();
            }
        };
        mapWebView.widthProperty().addListener(resizeListener);
        mapWebView.heightProperty().addListener(resizeListener);
    }

    private void refreshMap() {
        // Debounce/Delay to ensure layout is done
        new java.util.Timer().schedule(new java.util.TimerTask() {
            @Override
            public void run() {
                javafx.application.Platform.runLater(() -> {
                    mapWebView.getEngine().executeScript("if(window.map) { window.map.invalidateSize(); }");
                });
            }
        }, 100);
    }

    /**
     * Bridge class for JS -> Java communication.
     * Methods must be public.
     * Use static class to ensure clean JavaFX/Nashorn visibility
     */
    @Slf4j
    public static class JavaBridge {
        private final ApplicationEventPublisher eventPublisher;

        public JavaBridge(ApplicationEventPublisher eventPublisher) {
            this.eventPublisher = eventPublisher;
        }

        public void onStationSelected(String stationId) {
            log.info("Usuario seleccionó estación en mapa: {}", stationId);
            if (eventPublisher != null) {
                // We need to pass source as null or a new object since we are static
                // Or just use a dummy source. Ideally pass the controller, but static class...
                // Let's pass 'this' (JavaBridge instance) as source.
                eventPublisher.publishEvent(
                        new projectstalker.ui.event.StationSelectedEvent(this, stationId));
            } else {
                log.warn("EventPublisher is null, cannot publish selection.");
            }
        }

        public void logFromJs(String message) {
            log.info("[JS-MAP] {}", message);
        }

        public void errorFromJs(String message) {
            log.error("[JS-MAP-ERROR] {}", message);
        }
    }

    /**
     * Carga los datos CSV y JSON y los inyecta en el JS.
     */
    public void loadData() {
        if (!mapReady)
            return;

        // 1. Cargar Río Tajo (CSV)
        try {
            String wkt = loadResourceFile("maps/linestring_río_tajo.csv");
            String[] lines = wkt.split("\n");
            for (String line : lines) {
                if (line.trim().startsWith("\"MULTILINESTRING") || line.trim().startsWith("MULTILINESTRING")) {
                    injectRiver(line.trim().replace("\"", ""), "#00aaff"); // Azul Tajo
                }
            }
        } catch (Exception e) {
            log.error("Error cargando río Tajo", e);
        }

        // 2. Cargar Estaciones (JSON)
        try {
            String json = loadResourceFile("maps/saica_stations_maestro_coords.json");
            injectStations(json);
        } catch (Exception e) {
            log.error("Error cargando estaciones", e);
        }
    }

    private void injectRiver(String wkt, String color) {
        log.debug("Inyectando Río: {}", wkt.substring(0, Math.min(50, wkt.length())) + "...");
        String script = String.format("window.addRiverLineString('%s', '%s');", escapeJs(wkt), color);
        mapWebView.getEngine().executeScript(script);
    }

    private void injectStations(String json) {
        log.debug("Inyectando Estaciones...");
        String safeJson = json.replace("'", "\\'")
                .replace("\n", "")
                .replace("\r", "");

        String script = "window.addStations('" + safeJson + "');";
        mapWebView.getEngine().executeScript(script);
    }

    private String escapeJs(String text) {
        return text.replace("'", "\\'").replace("\n", "");
    }

    public void updatePopup(String stationId, String htmlContent) {
        Platform.runLater(() -> {
            try {
                // Escape quotes
                String safeHtml = htmlContent.replace("'", "\\'");
                mapWebView.getEngine()
                        .executeScript("window.updatePopupContent('" + stationId + "', '" + safeHtml + "');");
            } catch (Exception e) {
                log.error("Error updating popup for " + stationId, e);
            }
        });
    }

    public void focusStation(String stationId) {
        Platform.runLater(() -> {
            try {
                mapWebView.getEngine().executeScript("window.focusStation('" + stationId + "');");
            } catch (Exception e) {
                log.error("Error focusing station " + stationId, e);
            }
        });
    }

    public void injectBulkData(java.util.Map<String, String> dataMap) {
        if (dataMap == null || dataMap.isEmpty())
            return;

        if (!mapReady) {
            log.info("Map not ready yet. Queuing bulk data for later injection.");
            this.pendingBulkData = dataMap;
            return;
        }

        Platform.runLater(() -> {
            try {
                com.fasterxml.jackson.databind.ObjectMapper mapper = new com.fasterxml.jackson.databind.ObjectMapper();
                String json = mapper.writeValueAsString(dataMap);

                // Robust Injection: Pass data as a property instead of string concatenation
                netscape.javascript.JSObject window = (netscape.javascript.JSObject) mapWebView.getEngine()
                        .executeScript("window");
                window.setMember("tempBulkData", json);

                mapWebView.getEngine()
                        .executeScript("window.bulkUpdatePopups(window.tempBulkData); window.tempBulkData = null;");

            } catch (Exception e) {
                log.error("Error injecting bulk popup data", e);
            }
        });
    }

    private String loadResourceFile(String path) throws IOException {
        ClassPathResource resource = new ClassPathResource(path);
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(resource.getInputStream(), StandardCharsets.UTF_8))) {
            return reader.lines().collect(Collectors.joining("\n"));
        }
    }
}
