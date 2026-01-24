package projectstalker.compute.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import projectstalker.compute.entity.SimulationEntity;

@Repository
public interface SimulationRepository extends JpaRepository<SimulationEntity, String> {
}
