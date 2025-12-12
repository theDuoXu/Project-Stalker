package projectstalker.domain.dto.sensor;

import com.fasterxml.jackson.annotation.JsonView;
import lombok.Builder;
import lombok.With;
import projectstalker.domain.sensors.SensorViews;

@Builder
@With
@JsonView(SensorViews.Public.class)
public record SensorReadingDTO(
        String tag,
        String timestamp,
        Double value,
        String formattedValue,
        String stationId
) {}