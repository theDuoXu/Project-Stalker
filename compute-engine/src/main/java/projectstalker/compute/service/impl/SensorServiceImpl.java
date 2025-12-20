package projectstalker.compute.service.impl;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import projectstalker.compute.entity.SensorEntity;
import projectstalker.compute.repository.DigitalTwinRepository;
import projectstalker.compute.repository.SensorRepository;
import projectstalker.compute.service.SensorService;
import projectstalker.domain.dto.sensor.*;
import projectstalker.domain.exception.InvalidExportRequestException;
import projectstalker.domain.exception.SensorBusinessException;
import projectstalker.domain.sensors.*;

import java.time.Duration;
import java.time.LocalDateTime;
import java.util.List;

@Slf4j
@Service
@RequiredArgsConstructor
public class SensorServiceImpl implements SensorService {
    private final DigitalTwinRepository twinRepository;
    private final SensorRepository sensorRepository;

    @Override
    public SensorResponseDTO getHistory(String stationId, String parameter) {
        String paramNormalized = parameter.toUpperCase();
        SensorType sensorType = SensorType.fromString(parameter);

        if (sensorType == SensorType.UNKNOWN) {
            throw new IllegalArgumentException("Parameter: "+ paramNormalized +" does not match any known sensor type");
        }

        List<SensorReadingDTO> readings = sensorRepository.findReadings(stationId, paramNormalized);

        return SensorResponseDTO.builder()
                .stationId(stationId)
                .name(sensorType.getCode())
                .signalType(sensorType.getSignalType())
                .unit(sensorType.getUnit())
                .values(readings)
                .build();
    }

    @Override
    public SensorHealthResponseDTO getHealth(String stationId, String parameter) {
        String paramUpper = parameter.toUpperCase();
        List<SensorHealthDTO> healthData;

        if ("ALL".equals(paramUpper)) {
            healthData = sensorRepository.findHealthStatus(stationId);
        } else {
            SensorType type = SensorType.fromString(paramUpper);
            if (type == SensorType.UNKNOWN) {
                healthData = List.of();
            } else {
                healthData = sensorRepository.findHealthStatusByType(stationId, type);
            }
        }

        boolean isAllOk = healthData.stream()
                .allMatch(h -> (h.batteryPercentage() > 20) &&
                        (Duration.between(h.lastChecked(), LocalDateTime.now()).toHours() < 24));

        return SensorHealthResponseDTO.builder()
                .stationId(stationId)
                .isAllOk(isAllOk)
                .values(healthData)
                .build();
    }

    @Override
    public List<SensorReadingDTO> getRealtime(String stationId, String parameter) {
        String paramUpper = parameter.toUpperCase();

        if ("ALL".equals(paramUpper)) {
            return sensorRepository.findLatestReadings(stationId);
        }

        SensorType type = SensorType.fromString(paramUpper);
        if (type == SensorType.UNKNOWN) {
            return List.of();
        }
        return sensorRepository.findLatestReadingsByType(stationId, type);
    }

    @Override
    public SensorResponseDTO getExportData(String stationId, String parameter, LocalDateTime from, LocalDateTime to) {
        String paramUpper = parameter.toUpperCase();

        if ("ALL".equals(paramUpper)) {
            throw new SensorBusinessException("Bulk export for ALL parameters is restricted.");
        }

        SensorType type = SensorType.fromString(paramUpper);
        if (type == SensorType.UNKNOWN) {
            throw new InvalidExportRequestException("Unknown parameter: '" + parameter + "'.");
        }

        if (to == null) to = LocalDateTime.now();
        if (from == null) from = LocalDateTime.now().minusYears(100);

        List<SensorReadingDTO> readings = sensorRepository.findReadingsByDateRange(stationId, type.getCode(), from, to);

        return SensorResponseDTO.builder()
                .stationId(stationId)
                .name(type.getCode())
                .signalType(type.getSignalType())
                .unit(type.getUnit())
                .values(readings)
                .build();
    }

    @Override
    @Transactional
    public SensorResponseDTO registerSensor(SensorCreationDTO request) {
        log.info("Registrando nuevo sensor: {}", request.name());

        // 1. Validaciones
        SensorType type = SensorType.fromString(request.type());
        if (type == SensorType.UNKNOWN) throw new SensorBusinessException("Tipo inválido");

        // Necesitamos el ID del Gemelo al que pertenece (debería venir en el DTO)
        String twinId = request.configuration().getOrDefault("twinId", "").toString();

        // Recuperar la entidad del Río (Referencia JPA)
        // Usamos getReferenceById si solo queremos enlazar sin hacer SELECT,
        // pero findById es más seguro para validar que el río existe.
        var river = twinRepository.findById(twinId)
                .orElseThrow(() -> new SensorBusinessException("El Río (Twin) especificado no existe."));

        // 2. Construir Entidad
        SensorEntity entity = SensorEntity.builder()
                .name(request.name())
                .type(type)
                .locationKm(request.locationKm())
                .strategyType(request.strategyType())
                .configuration(request.configuration()) // JSONB automático
                .twin(river) // Relación FK
                .isActive(true)
                .build();

        // 3. Guardar (El ID UUID se genera aquí si no lo seteaste)
        SensorEntity saved = sensorRepository.save(entity);

        // 4. Retornar DTO con el ID generado
        return SensorResponseDTO.builder()
                .stationId(saved.getId()) // UUID real
                .name(saved.getName())
                .signalType(saved.getType().getSignalType())
                .unit(saved.getType().getUnit())
                .values(List.of())
                .build();
    }
}