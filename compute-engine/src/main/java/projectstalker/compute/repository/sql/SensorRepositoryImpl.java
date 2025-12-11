package projectstalker.compute.repository.sql;

import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Repository;
import projectstalker.compute.repository.SensorRepository;
import projectstalker.compute.service.SensorService;
import projectstalker.domain.sensors.SensorReadingDTO;

import java.util.List;

@Repository
@Profile("sql")
public class SensorRepositoryImpl implements SensorRepository {
    @Override
    public List<SensorReadingDTO> findReadings(String stationId, String parameter) {
        return List.of();
    }
}
