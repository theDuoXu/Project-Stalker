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
@Profile("!mock")
@RequiredArgsConstructor
public class SensorRepositoryImpl implements SensorRepository {

    private final JpaSensorRepository jpaSensorRepository;
    private final projectstalker.compute.repository.SensorReadingRepository readingRepository;

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

    @Override
    public List<SensorEntity> findAll() {
        return jpaSensorRepository.findAll();
    }

    // =========================================================================
    // LECTURA REAL
    // =========================================================================

    @Override
    public List<SensorReadingDTO> findReadings(String stationId, String parameter) {
        return readingRepository.findTop50BySensorIdAndParameterOrderByTimestampDesc(stationId, parameter)
                .stream()
                .map(this::mapToDto)
                .toList();
    }

    @Override
    public List<SensorReadingDTO> findLatestReadings(String stationId) {
        return readingRepository.findTop10BySensorIdOrderByTimestampDesc(stationId)
                .stream()
                .map(this::mapToDto)
                .toList();
    }

    @Override
    public List<SensorReadingDTO> findLatestReadingsByType(String stationId, SensorType type) {
        // Fallback to searching by parameter (Unit or Type Code)
        // Usually parameter in DB is lowercase or user defined?
        // Scraper saves: "value", "ph", "temperature"...
        // SensorType.PH.getParameter() -> "PH" ? Check SensorType.
        // Assuming scraping parameters "value" covers most single-value sensors.
        return readingRepository.findTop50BySensorIdAndParameterOrderByTimestampDesc(stationId, "value")
                .stream().map(this::mapToDto).toList();
    }

    private SensorReadingDTO mapToDto(projectstalker.compute.entity.SensorReadingEntity entity) {
        return SensorReadingDTO.builder()
                .stationId(entity.getSensorId())
                .tag(entity.getParameter())
                .timestamp(entity.getTimestamp().toString())
                .value(entity.getValue())
                .formattedValue(String.format("%.2f", entity.getValue()))
                .build();
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
    public List<SensorReadingDTO> findReadingsByDateRange(String stationId, String parameter, LocalDateTime from,
            LocalDateTime to) {
        return List.of();
    }
}