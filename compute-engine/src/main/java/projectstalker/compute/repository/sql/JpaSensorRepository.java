package projectstalker.compute.repository.sql;

import org.springframework.data.jpa.repository.JpaRepository;
import projectstalker.compute.entity.SensorEntity;

import java.util.List;

/**
 * Interfaz interna de Spring Data JPA.
 * SensorRepositoryImpl delegará en esta interfaz.
 */
public interface JpaSensorRepository extends JpaRepository<SensorEntity, String> {

    @org.springframework.data.jpa.repository.Query("SELECT s FROM SensorEntity s WHERE s.twin.id = :twinId")
    List<SensorEntity> findAllByTwinId(@org.springframework.data.repository.query.Param("twinId") String twinId);
}