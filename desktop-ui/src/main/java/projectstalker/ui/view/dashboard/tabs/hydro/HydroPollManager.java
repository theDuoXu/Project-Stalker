package projectstalker.ui.view.dashboard.tabs.hydro;

import javafx.animation.Animation;
import javafx.animation.KeyFrame;
import javafx.animation.Timeline;
import javafx.util.Duration;
import lombok.extern.slf4j.Slf4j;
import projectstalker.domain.dto.sensor.SensorReadingDTO;
import projectstalker.ui.service.SensorClientService;

import java.util.List;
import java.util.function.Consumer;

@Slf4j
public class HydroPollManager {

    private final SensorClientService sensorService;
    private Timeline activePoll;

    public HydroPollManager(SensorClientService sensorService) {
        this.sensorService = sensorService;
    }

    public void startPolling(String stationId, Consumer<List<SensorReadingDTO>> onDataReceived) {
        stopPolling();

        activePoll = new Timeline(new KeyFrame(Duration.seconds(4), ev -> {
            sensorService.getRealtime(stationId, "ALL")
                    .collectList()
                    .subscribe(readings -> {
                        if (readings != null && !readings.isEmpty()) {
                            onDataReceived.accept(readings);

                            // Heuristic: If we get bulk history suddenly, likely no need to poll fast?
                            // Keep simple for now.
                        }
                    }, err -> log.error("HydroPollManager poll error: {}", err.getMessage()));
        }));

        activePoll.setCycleCount(Animation.INDEFINITE);
        activePoll.play();
        log.info("Started polling for station: {}", stationId);
    }

    public void stopPolling() {
        if (activePoll != null) {
            activePoll.stop();
            activePoll = null;
            log.debug("Polling stopped.");
        }
    }

    public boolean isPolling() {
        return activePoll != null && activePoll.getStatus() == Animation.Status.RUNNING;
    }
}
