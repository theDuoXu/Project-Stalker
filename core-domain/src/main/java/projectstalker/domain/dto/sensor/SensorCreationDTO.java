package projectstalker.domain.dto.sensor;

import java.util.Map;

public record SensorCreationDTO(
                String name,
                String type,
                double locationKm,
                String strategyType, // "VIRTUAL" o "REAL"
                String twinId, // Explicit Twin Context
                Map<String, Object> configuration // Payload flexible
) {
}