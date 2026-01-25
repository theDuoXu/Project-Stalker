package projectstalker.ui.view.dashboard.tabs;

import javafx.concurrent.Worker;
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
public class LeafletMapController {

    @Autowired
    private ApplicationEventPublisher eventPublisher;

    @FXML
    private WebView mapWebView;

    private boolean mapReady = false;

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

        // Listener de carga
        engine.getLoadWorker().stateProperty().addListener((obs, oldState, newState) -> {
            if (newState == Worker.State.SUCCEEDED) {
                log.info("Mapa cargado correctamente en WebView.");

                // Inject Java Bridge (Interaction)
                netscape.javascript.JSObject window = (netscape.javascript.JSObject) engine.executeScript("window");
                window.setMember("javaConnector", new JavaBridge());

                mapReady = true;
                loadData();
            } else if (newState == Worker.State.FAILED) {
                log.error("Fallo al cargar el mapa HTML.");
            }
        });

        // Resize Listener: Force map update when window/webview resizing
        mapWebView.widthProperty().addListener((obs, oldVal, newVal) -> {
            if (mapReady) {
                engine.executeScript("if(window.map) { window.map.invalidateSize(); }");
            }
        });
        mapWebView.heightProperty().addListener((obs, oldVal, newVal) -> {
            if (mapReady) {
                engine.executeScript("if(window.map) { window.map.invalidateSize(); }");
            }
        });
    }

    /**
     * Bridge class for JS -> Java communication.
     * Methods must be public.
     */
    public class JavaBridge {
        public void onStationSelected(String stationId) {
            log.info("Usuario seleccionó estación en mapa: {}", stationId);
            if (eventPublisher != null) {
                eventPublisher.publishEvent(
                        new projectstalker.ui.event.StationSelectedEvent(LeafletMapController.this, stationId));
            } else {
                log.warn("EventPublisher is null, cannot publish selection.");
            }
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

    private String loadResourceFile(String path) throws IOException {
        ClassPathResource resource = new ClassPathResource(path);
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(resource.getInputStream(), StandardCharsets.UTF_8))) {
            return reader.lines().collect(Collectors.joining("\n"));
        }
    }
}
