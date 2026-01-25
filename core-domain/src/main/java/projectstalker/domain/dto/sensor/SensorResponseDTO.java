package projectstalker.domain.dto.sensor;

import com.fasterxml.jackson.annotation.JsonView;
import lombok.Builder;
import lombok.With;
import projectstalker.domain.sensors.SensorViews;

import java.util.List;

@Builder
@With
@JsonView(SensorViews.Public.class)
public record SensorResponseDTO(
                String stationId,
                String name,
                String signalType,
                String unit,
                List<SensorReadingDTO> values,
                // New fields for UI Editing
                java.util.Map<String, Object> configuration,
                String typeCode) {
}