package projectstalker.domain.sensors;

import lombok.Builder;

@Builder
public record SensorReadingDTO(
        String tag,
        String timestamp,
        Double value,
        String formattedValue,
        String stationId
) {}