package projectstalker.compute.service.impl;

import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import projectstalker.compute.repository.SensorRepository;
import projectstalker.compute.service.SensorService;
import projectstalker.domain.dto.sensor.SensorHealthDTO;
import projectstalker.domain.dto.sensor.SensorHealthResponseDTO;
import projectstalker.domain.dto.sensor.SensorReadingDTO;
import projectstalker.domain.dto.sensor.SensorResponseDTO;
import projectstalker.domain.exception.InvalidExportRequestException;
import projectstalker.domain.exception.SensorBusinessException;
import projectstalker.domain.sensors.*;

import java.time.Duration;
import java.time.LocalDateTime;
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

    @Override
    public SensorHealthResponseDTO getHealth(String stationId, String parameter) {
        String paramUpper = parameter.toUpperCase();
        List<SensorHealthDTO> healthData;

        // 1. OBTENER DATOS DEL REPO
        if ("ALL".equals(paramUpper)) {
            healthData = sensorRepository.findHealthStatus(stationId);
        } else {
            SensorType type = SensorType.fromString(paramUpper);
            if (type == SensorType.UNKNOWN) {
                // Manejo de error o lista vacía
                healthData = List.of();
            } else {
                healthData = sensorRepository.findHealthStatusByType(stationId, type);
            }
        }

        // 2. LÓGICA DE NEGOCIO: CALCULAR isAllOk
        // Consideramos que está OK si la batería > 20%
        // Y lastChecked menos de un día
        boolean isAllOk =
                healthData.stream()
                .allMatch(h ->
                        (h.batteryPercentage() > 20) &&
                        (Duration.between(h.lastChecked(), LocalDateTime.now()).toHours() < 24));

        // 3. CONSTRUIR RESPUESTA
        return SensorHealthResponseDTO.builder()
                .stationId(stationId)
                .isAllOk(isAllOk)
                .values(healthData)
                .build();
    }

    @Override
    public List<SensorReadingDTO> getRealtime(String stationId, String parameter) {
        String paramUpper = parameter.toUpperCase();

        // CASO A: El usuario quiere todo
        if ("ALL".equals(paramUpper)) {
            return sensorRepository.findLatestReadings(stationId);
        }

        // CASO B: El usuario quiere un parámetro específico
        SensorType type = SensorType.fromString(paramUpper);
        if (type == SensorType.UNKNOWN) {
            // Opción: Devolver lista vacía, es más seguro para realtime.
            return List.of();
        }
        return sensorRepository.findLatestReadingsByType(stationId, type);
    }

    @Override
    public SensorResponseDTO getExportData(String stationId, String parameter, LocalDateTime from, LocalDateTime to) {
        String paramUpper = parameter.toUpperCase();

        // 1. CLÁUSULA DE PROTECCIÓN: Prohibido "ALL"
        // Protegemos el ancho de banda y la base de datos.
        if ("ALL".equals(paramUpper)) {
            throw new SensorBusinessException("Bulk export for ALL parameters is restricted. Please request specific metrics individually.");
        }

        // 2. VALIDACIÓN DE TIPO (Strict Mode)
        // Para un export científico, necesitamos saber la unidad exacta. Si es desconocido, no exportamos.
        SensorType type = SensorType.fromString(paramUpper);
        if (type == SensorType.UNKNOWN) {
            throw new InvalidExportRequestException("Unknown parameter: '" + parameter + "'. Cannot generate export for undefined metrics.");
        }

        // 3. Normalización de Fechas
        if (to == null) to = LocalDateTime.now();
        if (from == null) from = LocalDateTime.now().minusYears(100);

        // 4. Obtener Datos del Repositorio
        // Ahora es seguro llamar al repo porque sabemos que type es válido y único.
        List<SensorReadingDTO> readings = sensorRepository.findReadingsByDateRange(stationId, type.getCode(), from, to);

        // 5. Empaquetar con Metadatos Garantizados
        return SensorResponseDTO.builder()
                .name(type.getCode())
                .signalType(type.getSignalType())
                .unit(type.getUnit())
                .values(readings)
                .build();
    }
}