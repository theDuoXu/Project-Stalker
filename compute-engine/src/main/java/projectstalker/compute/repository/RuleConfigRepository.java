package projectstalker.compute.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import projectstalker.compute.entity.RuleConfigEntity;

import java.util.Optional;

@Repository
public interface RuleConfigRepository extends JpaRepository<RuleConfigEntity, Long> {
    Optional<RuleConfigEntity> findByMetric(String metric);
}
