package projectstalker.compute.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import projectstalker.compute.entity.DigitalTwinEntity;

import java.util.Optional;

/**
 * Repositorio para la gestión de persistencia de Gemelos Digitales.
 * <p>
 * Extiende de {@link JpaRepository} para obtener operaciones CRUD completas
 * y soporte de paginación (PagingAndSortingRepository) de forma nativa.
 * <p>
 * La implementación subyacente utiliza el DataSource configurado (HikariCP)
 * para gestionar eficientemente las conexiones a PostgreSQL.
 */
@Repository
public interface DigitalTwinRepository extends JpaRepository<DigitalTwinEntity, String> {

    /**
     * Verifica si ya existe un río con ese nombre (para evitar duplicados lógicos).
     */
    boolean existsByName(String name);

    // findAll(Pageable) y findById(String) ya vienen incluidos en JpaRepository.
    // No hace falta declararlos explícitamente.
}