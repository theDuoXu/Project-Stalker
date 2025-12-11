package projectstalker.compute.repository;

import projectstalker.domain.sensors.SensorReadingDTO;
import java.util.List;

public interface SensorRepository {
    /**
     * Recupera el histórico de datos, sin importar de dónde vengan.
     */
    List<SensorReadingDTO> findReadings(String stationId, String parameter);
}