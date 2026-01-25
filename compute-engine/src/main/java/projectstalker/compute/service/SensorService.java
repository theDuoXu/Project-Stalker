package projectstalker.compute.service;

import projectstalker.domain.dto.sensor.SensorCreationDTO;
import projectstalker.domain.dto.sensor.SensorHealthResponseDTO;
import projectstalker.domain.dto.sensor.SensorReadingDTO;
import projectstalker.domain.dto.sensor.SensorResponseDTO;

import java.time.LocalDateTime;
import java.util.List;

public interface SensorService {
    /**
     * Obtiene el histórico de mediciones de un parámetro específico en una
     * estación.
     * 
     * @param stationId El ID de la estación (ej: C302)
     * @param parameter El parámetro a medir (ej: AMONIO)
     * @return El DTO con la respuesta estructurada
     */
    SensorResponseDTO getHistory(String stationId, String parameter);

    SensorHealthResponseDTO getHealth(String stationId, String parameter);

    List<SensorReadingDTO> getRealtime(String stationId, String parameter);

    public SensorResponseDTO getExportData(String stationId, String parameter, LocalDateTime from, LocalDateTime to);

    SensorResponseDTO registerSensor(SensorCreationDTO request);

    SensorResponseDTO updateSensor(String stationId, SensorCreationDTO request);

    List<SensorResponseDTO> getAllByTwin(String twinId);
}