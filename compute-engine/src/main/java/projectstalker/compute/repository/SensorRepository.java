package projectstalker.compute.repository;

import projectstalker.domain.dto.sensor.SensorHealthDTO;
import projectstalker.domain.dto.sensor.SensorReadingDTO;
import projectstalker.domain.sensors.SensorType;

import java.time.LocalDateTime;
import java.util.List;

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
     * Devuelve la salud de un sensor específico (en una lista de 1 elemento para consistencia).
     */
    List<SensorHealthDTO> findHealthStatusByType(String stationId, SensorType type);

    List<SensorReadingDTO> findReadingsByDateRange(String stationId, String parameter, LocalDateTime from, LocalDateTime to);

}