package projectstalker.compute.repository.scraper;

import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Repository;
import projectstalker.compute.repository.SensorRepository;
import projectstalker.domain.sensors.SensorReadingDTO;

import java.util.List;

@Repository
@Profile("scraper")
public class SensorScraper implements SensorRepository {

    @Override
    public List<SensorReadingDTO> findReadings(String stationId, String parameter) {
        return List.of();
    }
}
