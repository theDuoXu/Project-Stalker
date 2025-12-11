package projectstalker.compute.service;

import projectstalker.domain.sensors.SensorResponseDTO;

public interface SensorService {
    /**
     * Obtiene el histórico de mediciones de un parámetro específico en una estación.
     * @param stationId El ID de la estación (ej: C302)
     * @param parameter El parámetro a medir (ej: AMONIO)
     * @return El DTO con la respuesta estructurada
     */
    SensorResponseDTO getHistory(String stationId, String parameter);
}