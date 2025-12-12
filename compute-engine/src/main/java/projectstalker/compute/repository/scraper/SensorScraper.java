package projectstalker.compute.repository.scraper;

import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Repository;
import projectstalker.compute.repository.SensorRepository;
import projectstalker.domain.dto.sensor.SensorHealthDTO;
import projectstalker.domain.dto.sensor.SensorReadingDTO;
import projectstalker.domain.sensors.SensorType;

import java.time.LocalDateTime;
import java.util.List;

@Repository
@Profile("scraper")
public class SensorScraper implements SensorRepository {

    @Override
    public List<SensorReadingDTO> findReadings(String stationId, String parameter) {
        return List.of();
    }

    @Override
    public List<SensorReadingDTO> findLatestReadings(String stationId) {
        return List.of();
    }

    @Override
    public List<SensorReadingDTO> findLatestReadingsByType(String stationId, SensorType type) {
        return List.of();
    }

    @Override
    public List<SensorHealthDTO> findHealthStatus(String stationId) {
        return List.of();
    }

    @Override
    public List<SensorHealthDTO> findHealthStatusByType(String stationId, SensorType type) {
        return List.of();
    }

    @Override
    public List<SensorReadingDTO> findReadingsByDateRange(String stationId, String parameter, LocalDateTime from, LocalDateTime to) {
        return List.of();
    }
}
