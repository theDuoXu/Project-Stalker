package projectstalker.ui.view.dashboard.tabs;

import javafx.concurrent.Worker;
import javafx.fxml.FXML;
import javafx.scene.web.WebEngine;
import javafx.scene.web.WebView;
import lombok.extern.slf4j.Slf4j;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Component;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.stream.Collectors;

@Slf4j
@Component
public class LeafletMapController {

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
                mapReady = true;
                loadData();
            } else if (newState == Worker.State.FAILED) {
                log.error("Fallo al cargar el mapa HTML.");
            }
        });
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
            // El CSV tiene una cabecera "geometry", la quitamos si existe o cogemos la
            // segunda linea
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
        // Escapamos comillas simples para JS
        String safeJson = json.replace("'", "\\'");
        // Nota: executeScript puede fallar si el string es muy largo en algunas
        // versiones de JavaFX.
        // Si falla, habría que asignarlo a una variable JS en trozos. Pero para 11KB
        // debería ir bien.
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
