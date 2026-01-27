package projectstalker.compute.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import projectstalker.compute.entity.SensorReadingEntity;

import java.time.LocalDateTime;
import java.util.List;

@Repository
public interface SensorReadingRepository extends JpaRepository<SensorReadingEntity, String> {
        List<SensorReadingEntity> findBySensorIdAndTimestampAfter(String sensorId, LocalDateTime timestamp);

        List<SensorReadingEntity> findTop50BySensorIdAndParameterOrderByTimestampDesc(String sensorId,
                        String parameter);

        List<SensorReadingEntity> findTop10BySensorIdOrderByTimestampDesc(String sensorId);

        // Dynamic Limit using Pageable
        List<SensorReadingEntity> findBySensorIdAndParameterOrderByTimestampDesc(String sensorId, String parameter,
                        org.springframework.data.domain.Pageable pageable);

        List<SensorReadingEntity> findBySensorIdAndParameterAndTimestampAfter(String sensorId, String parameter,
                        LocalDateTime timestamp);

        List<SensorReadingEntity> findBySensorIdAndParameterIgnoreCaseAndTimestampAfter(String sensorId,
                        String parameter,
                        LocalDateTime timestamp);

        List<SensorReadingEntity> findBySensorIdAndParameterIgnoreCaseOrderByTimestampDesc(String sensorId,
                        String parameter,
                        org.springframework.data.domain.Pageable pageable);

        List<SensorReadingEntity> findBySensorIdAndTimestampBetween(String sensorId, LocalDateTime start,
                        LocalDateTime end);

        @org.springframework.data.jpa.repository.Query("SELECT DISTINCT r.sensorId FROM SensorReadingEntity r")
        List<String> findDistinctSensorIds();
}
