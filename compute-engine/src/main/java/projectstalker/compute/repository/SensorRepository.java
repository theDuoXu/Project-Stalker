package projectstalker.compute.repository;

import projectstalker.compute.entity.SensorEntity;
import projectstalker.domain.dto.sensor.SensorHealthDTO;
import projectstalker.domain.dto.sensor.SensorReadingDTO;
import projectstalker.domain.sensors.SensorType;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

public interface SensorRepository {
    /**
     * Recupera el histórico de datos, sin importar de dónde vengan.
     */
    List<SensorReadingDTO> findReadings(String stationId, String parameter);

    /**
     * Responde a get realtime ALL
     */
    List<SensorReadingDTO> findLatestReadings(String stationId);

    /**
     * Responde a get realtime de un sensor específico
     */
    List<SensorReadingDTO> findLatestReadingsByType(String stationId, SensorType type);

    /**
     * Devuelve la salud de TODOS los sensores de la estación.
     */
    List<SensorHealthDTO> findHealthStatus(String stationId);

    /**
     * Devuelve la salud de un sensor específico (en una lista de 1 elemento para
     * consistencia).
     */
    List<SensorHealthDTO> findHealthStatusByType(String stationId, SensorType type);

    List<SensorReadingDTO> findReadingsByDateRange(String stationId, String parameter, LocalDateTime from,
            LocalDateTime to);

    boolean existsById(String id);

    /**
     * Guardar sensor completo
     */
    SensorEntity save(SensorEntity sensor);

    /**
     * Buscar todos los sensores de un río específico
     */
    List<SensorEntity> findAllByTwinId(String twinId);

    /**
     * Buscar un sensor y devolver la entidad (útil para verificar antes de
     * borrar/editar)
     */
    Optional<SensorEntity> findById(String id);

    /**
     * Buscar todos los sensores registrados (para resolución interna de IDs)
     */
    List<SensorEntity> findAll();

}