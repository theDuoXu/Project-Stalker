package projectstalker.compute.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import projectstalker.compute.entity.AlertEntity;

import java.time.LocalDateTime;
import java.util.List;

@Repository
public interface AlertRepository extends JpaRepository<AlertEntity, String> {
    List<AlertEntity> findBySensorId(String sensorId);

    List<AlertEntity> findByStatus(AlertEntity.AlertStatus status);

    List<AlertEntity> findByStatusIn(java.util.Collection<AlertEntity.AlertStatus> statuses);

    List<AlertEntity> findByTimestampAfter(LocalDateTime timestamp);

    boolean existsBySensorIdAndMetricAndTimestampAfter(String sensorId, String metric, LocalDateTime timestamp);

    org.springframework.data.domain.Page<AlertEntity> findByTimestampBetween(LocalDateTime start, LocalDateTime end,
            org.springframework.data.domain.Pageable pageable);

    org.springframework.data.domain.Page<AlertEntity> findByTimestampBetweenAndStatusIn(LocalDateTime start,
            LocalDateTime end,
            java.util.Collection<AlertEntity.AlertStatus> statuses,
            org.springframework.data.domain.Pageable pageable);
}
