package projectstalker.compute.service.impl;

import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import projectstalker.compute.repository.SensorRepository;
import projectstalker.compute.service.SensorService;
import projectstalker.domain.sensors.SensorReadingDTO;
import projectstalker.domain.sensors.SensorResponseDTO;
import projectstalker.domain.sensors.SensorType; // <-- 1. Importar el enum

import java.util.List;

@Service
@RequiredArgsConstructor
public class SensorServiceImpl implements SensorService {

    private final SensorRepository sensorRepository;

    @Override
    public SensorResponseDTO getHistory(String stationId, String parameter) {
        String paramNormalized = parameter.toUpperCase();

        // 1. Obtener el tipo de sensor usando el enum (proporciona validación implícita)
        SensorType sensorType = SensorType.fromString(parameter);

        // Manejo de errores si el tipo es UNKNOWN
        if (sensorType == SensorType.UNKNOWN) {
            throw new IllegalArgumentException("Parameter: "+ paramNormalized +" does not match any known sensor type");
        }

        // 2. LLAMADA AL REPOSITORIO (Agnóstico de dónde vienen los datos)
        // Se sigue usando el String normalizado para la consulta (el repositorio usa strings)
        List<SensorReadingDTO> readings = sensorRepository.findReadings(stationId, paramNormalized);

        // 3. CONSTRUCCIÓN DE LA RESPUESTA - USANDO EL ENUM
        return SensorResponseDTO.builder()
                .name(sensorType.getCode()) // Usar el code del enum
                .signalType(sensorType.getSignalType()) // Obtenido del enum
                .unit(sensorType.getUnit()) // Obtenido del enum
                .values(readings)
                .build();
    }
}