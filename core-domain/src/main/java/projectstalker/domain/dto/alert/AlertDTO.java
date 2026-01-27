package projectstalker.domain.dto.alert;

import java.time.LocalDateTime;

public record AlertDTO(
        String id,
        @com.fasterxml.jackson.annotation.JsonAlias("sensorId") String stationId,
        String stationName,
        AlertSeverity severity,
        AlertStatus status,
        String message,
        LocalDateTime timestamp,
        double value,
        String metric) {
}
