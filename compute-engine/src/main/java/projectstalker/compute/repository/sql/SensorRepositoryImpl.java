package projectstalker.compute.repository.sql;

import lombok.RequiredArgsConstructor;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Repository;
import projectstalker.compute.entity.SensorEntity;
import projectstalker.compute.repository.SensorRepository;
import projectstalker.domain.dto.sensor.SensorHealthDTO;
import projectstalker.domain.dto.sensor.SensorReadingDTO;
import projectstalker.domain.sensors.SensorType;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Repository
@Profile("sql")
@RequiredArgsConstructor
public class SensorRepositoryImpl implements SensorRepository {

    private final JpaSensorRepository jpaSensorRepository;

    // =========================================================================
    // IMPLEMENTACIÓN DE ESCRITURA (NUEVO)
    // =========================================================================

    @Override
    public boolean existsById(String id) {
        return jpaSensorRepository.existsById(id);
    }

    @Override
    public SensorEntity save(SensorEntity sensor) {
        return jpaSensorRepository.save(sensor);
    }

    @Override
    public List<SensorEntity> findAllByTwinId(String twinId) {
        return jpaSensorRepository.findAllByTwinId(twinId);
    }

    @Override
    public Optional<SensorEntity> findById(String id) {
        return jpaSensorRepository.findById(id);
    }

    // =========================================================================
    // STUBS DE LECTURA (Mantienen compatibilidad hasta que exista ReadingEntity)
    // =========================================================================

    @Override
    public List<SensorReadingDTO> findReadings(String stationId, String parameter) {
        // TODO: Implementar consulta a tabla de lecturas (TimeSeries)
        return List.of();
    }

    @Override
    public List<SensorReadingDTO> findLatestReadings(String stationId) {
        return List.of();
    }

    @Override
    public List<SensorReadingDTO> findLatestReadingsByType(String stationId, SensorType type) {
        return List.of();
    }

    @Override
    public List<SensorHealthDTO> findHealthStatus(String stationId) {
        return List.of();
    }

    @Override
    public List<SensorHealthDTO> findHealthStatusByType(String stationId, SensorType type) {
        return List.of();
    }

    @Override
    public List<SensorReadingDTO> findReadingsByDateRange(String stationId, String parameter, LocalDateTime from, LocalDateTime to) {
        return List.of();
    }
}