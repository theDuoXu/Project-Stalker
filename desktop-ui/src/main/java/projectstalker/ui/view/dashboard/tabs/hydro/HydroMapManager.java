package projectstalker.ui.view.dashboard.tabs.hydro;

import javafx.application.Platform;
import javafx.scene.Node;
import javafx.scene.control.ToggleButton;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import projectstalker.domain.dto.sensor.SensorReadingDTO;
import projectstalker.domain.dto.sensor.SensorResponseDTO;
import projectstalker.ui.service.SensorClientService;
import projectstalker.ui.view.dashboard.tabs.LeafletMapController;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Slf4j
@RequiredArgsConstructor
public class HydroMapManager {

    private final SensorClientService sensorService;
    private final LeafletMapController mapController;

    private Node chartNode;
    private Node toolbarNode;
    private Node mapNode;
    private ToggleButton viewToggle;

    public void setupViewToggle(ToggleButton viewToggle, Node chartNode, Node toolbarNode, Node mapNode) {
        this.viewToggle = viewToggle;
        this.chartNode = chartNode;
        this.toolbarNode = toolbarNode;
        this.mapNode = mapNode;

        if (viewToggle != null) {
            viewToggle.selectedProperty().addListener((obs, oldVal, newVal) -> {
                boolean showMap = newVal;
                updateVisibility(showMap);
            });
        }
    }

    private void updateVisibility(boolean showMap) {
        // Toggle Visibility
        if (chartNode != null) {
            chartNode.setVisible(!showMap);
            chartNode.setManaged(!showMap);
        }
        if (toolbarNode != null) {
            toolbarNode.setVisible(!showMap);
            toolbarNode.setManaged(!showMap);
        }

        if (mapNode != null) {
            mapNode.setVisible(showMap);
            mapNode.setManaged(showMap);
        }

        if (viewToggle != null) {
            viewToggle.setText(showMap ? "Ver Gráfica de Datos" : "Ver Mapa Geoespacial");
            org.kordamp.ikonli.javafx.FontIcon icon = (org.kordamp.ikonli.javafx.FontIcon) viewToggle.getGraphic();
            if (icon != null)
                icon.setIconLiteral(showMap ? "mdi2c-chart-line" : "mdi2m-map");
        }
    }

    public void focusStation(String stationId) {
        if (mapController != null && mapNode != null && mapNode.isVisible()) {
            mapController.focusStation(stationId);
        }
    }

    public void updatePopup(String stationId, List<SensorReadingDTO> readings) {
        if (mapNode != null && mapNode.isVisible() && mapController != null && !readings.isEmpty()) {
            String html = generatePopupHtml(stationId, readings);
            mapController.updatePopup(stationId, html);
        }
    }

    public void preloadData() {
        log.info("Iniciando precarga masiva de datos de estaciones (Conexión a Backend)...");

        sensorService.getAllAvailableSensors()
                .map(SensorResponseDTO::stationId)
                .distinct()
                .flatMap(stationId ->
                // 2. Para cada estación, pedir datos REALES al backend
                sensorService.getRealtime(stationId, "ALL")
                        .collectList()
                        .map(readings -> new AbstractMap.SimpleEntry<>(stationId, readings))
                        // Si falla una, no detener todo
                        .onErrorResume(e -> {
                            log.warn("Fallo precarga para {}: {}", stationId, e.getMessage());
                            return reactor.core.publisher.Mono
                                    .just(new AbstractMap.SimpleEntry<>(stationId, new ArrayList<>()));
                        }), 5) // Concurrencia controlada
                .collectList()
                .subscribe(results -> {
                    log.info("Precarga: Recibidos datos de {} estaciones del backend.", results.size());
                    Map<String, String> popupCache = new HashMap<>();

                    for (Map.Entry<String, List<SensorReadingDTO>> entry : results) {
                        String stationId = entry.getKey();
                        List<SensorReadingDTO> readings = entry.getValue();

                        if (readings == null || readings.isEmpty()) {
                            continue;
                        }

                        // 3. Agregar todas las lecturas (último valor de cada tag)
                        popupCache.put(stationId, generatePopupHtml(stationId, readings));
                    }

                    log.info("Precarga finalizada. Generado cache para {} estaciones. Inyectando...",
                            popupCache.size());
                    Platform.runLater(() -> {
                        if (mapController != null) {
                            mapController.injectBulkData(popupCache);
                        }
                    });
                }, err -> log.error("Error crítico en precarga masiva", err));
    }

    private String generatePopupHtml(String stationId, List<SensorReadingDTO> readings) {
        if (readings == null || readings.isEmpty())
            return "";

        // Unify logic from Controller
        Map<String, SensorReadingDTO> latestByTag = new HashMap<>();
        for (SensorReadingDTO r : readings) {
            latestByTag.merge(r.tag(), r,
                    (oldVal, newVal) -> oldVal.timestamp().compareTo(newVal.timestamp()) > 0 ? oldVal : newVal);
        }

        StringBuilder html = new StringBuilder();
        html.append("<div style='min-width:150px'>");
        html.append("<h4>").append(stationId).append("</h4>");
        html.append("<table style='width:100%; font-size:12px; border-collapse: collapse;'>");

        List<SensorReadingDTO> sorted = new ArrayList<>(latestByTag.values());
        sorted.sort(Comparator.comparing(SensorReadingDTO::tag));

        for (SensorReadingDTO r : sorted) {
            html.append("<tr>");
            html.append("<td style='color:#aad;'>").append(r.tag()).append(":</td>");
            html.append("<td style='text-align:right; font-weight:bold;'>")
                    .append(String.format("%.2f", r.value())).append("</td>");
            html.append("</tr>");
        }
        html.append("</table>");

        // Footer time
        if (!sorted.isEmpty()) {
            String ts = sorted.get(0).timestamp().replace("T", " ");
            if (ts.length() > 16)
                ts = ts.substring(5, 16); // mm-dd HH:mm
            html.append("<div style='font-size:10px; color:#888; margin-top:5px; text-align:right;'>")
                    .append(ts)
                    .append("</div>");
        }
        html.append("</div>");
        return html.toString();
    }
}
