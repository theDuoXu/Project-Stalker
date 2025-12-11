package projectstalker.domain.sensors;

import lombok.Builder;
import java.util.List;

@Builder
public record SensorResponseDTO(
        String name,
        String signalType,
        String unit,
        List<SensorReadingDTO> values
) {
}