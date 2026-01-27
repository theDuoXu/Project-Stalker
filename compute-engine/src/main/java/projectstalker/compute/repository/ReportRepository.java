package projectstalker.compute.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import projectstalker.compute.entity.ReportEntity;

@Repository
public interface ReportRepository extends JpaRepository<ReportEntity, String> {
}
