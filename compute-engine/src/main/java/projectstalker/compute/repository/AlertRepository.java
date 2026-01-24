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

    List<AlertEntity> findByTimestampAfter(LocalDateTime timestamp);
}
