package projectstalker.ui.view.dashboard.tabs.hydro;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import projectstalker.domain.dto.sensor.SensorResponseDTO;
import projectstalker.ui.service.SensorClientService;

import java.time.LocalDateTime;

@Slf4j
@RequiredArgsConstructor
public class HydroHistoryManager {

    private final SensorClientService sensorService;

    public reactor.core.publisher.Mono<SensorResponseDTO> fetchHistory(String stationId, String metric) {
        // Fetch last 10 days by default (or whatever default range makes sense now
        // given backend fetches all available)
        LocalDateTime to = LocalDateTime.now();
        LocalDateTime from = to.minusDays(10);

        log.info("Loading history for {} from {} to {}", stationId, from, to);
        if (metric == null || metric.isBlank()) {
            return reactor.core.publisher.Mono
                    .error(new IllegalArgumentException("Metric must be selected for history."));
        }
        return sensorService.getExportData(stationId, metric, from, to);
    }
}
