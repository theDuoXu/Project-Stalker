package projectstalker.domain.dto.alert;

import java.time.LocalDateTime;

import com.fasterxml.jackson.annotation.JsonView;
import projectstalker.domain.sensors.SensorViews;

@JsonView(SensorViews.Public.class)
public record AlertDTO(
        String id,
        @com.fasterxml.jackson.annotation.JsonAlias("sensorId") String stationId,
        String stationName,
        AlertSeverity severity,
        AlertStatus status,
        String message,
        LocalDateTime timestamp,
        double value,
        String metric,
        String reportId) {
}
