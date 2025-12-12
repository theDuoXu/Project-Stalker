package projectstalker.domain.dto.sensor;

import com.fasterxml.jackson.annotation.JsonView;
import lombok.Builder;
import lombok.With;
import projectstalker.domain.sensors.SensorViews;

import java.util.List;

@Builder
@With
@JsonView(SensorViews.Public.class)
public record SensorHealthResponseDTO(
        String stationId,
        boolean isAllOk,
        List<SensorHealthDTO> values
) {
}