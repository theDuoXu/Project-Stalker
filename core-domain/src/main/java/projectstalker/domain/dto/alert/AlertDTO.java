package projectstalker.domain.dto.alert;

import java.time.LocalDateTime;

public record AlertDTO(
                String id,
                String stationId,
                String stationName,
                AlertSeverity severity,
                AlertStatus status,
                String message,
                LocalDateTime timestamp,
                double value,
                String metric) {
}
