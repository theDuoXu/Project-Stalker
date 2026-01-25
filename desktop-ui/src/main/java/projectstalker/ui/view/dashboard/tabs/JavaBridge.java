package projectstalker.ui.view.dashboard.tabs;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.ApplicationEventPublisher;

/**
 * Bridge class for JS -> Java communication.
 * Extracted to top-level class to ensure visibility for JavaFX WebView.
 */
@Slf4j
@RequiredArgsConstructor
public class JavaBridge {

    private final ApplicationEventPublisher eventPublisher;

    public void onStationSelected(String stationId) {
        log.info("Usuario seleccionó estación en mapa: {}", stationId);
        if (eventPublisher != null) {
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
