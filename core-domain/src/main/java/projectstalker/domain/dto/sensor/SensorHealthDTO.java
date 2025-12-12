package projectstalker.domain.dto.sensor;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonView;
import lombok.Builder;
import lombok.With;
import projectstalker.domain.sensors.SensorViews;

import java.time.LocalDateTime;

@Builder
@With
@JsonView(SensorViews.Public.class)
public record SensorHealthDTO(

        String tag,
        @JsonFormat(shape = JsonFormat.Shape.STRING, pattern = "dd/MM/yyyy HH:mm:ss")
        LocalDateTime lastChecked,

        @JsonView(SensorViews.Internal.class)
        int batteryPercentage
) {}