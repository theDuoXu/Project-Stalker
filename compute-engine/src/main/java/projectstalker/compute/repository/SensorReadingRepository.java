package projectstalker.compute.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import projectstalker.compute.entity.SensorReadingEntity;

import java.time.LocalDateTime;
import java.util.List;

@Repository
public interface SensorReadingRepository extends JpaRepository<SensorReadingEntity, String> {
    List<SensorReadingEntity> findBySensorIdAndTimestampAfter(String sensorId, LocalDateTime timestamp);

    List<SensorReadingEntity> findTop50BySensorIdAndParameterOrderByTimestampDesc(String sensorId, String parameter);

    List<SensorReadingEntity> findTop10BySensorIdOrderByTimestampDesc(String sensorId);
}
