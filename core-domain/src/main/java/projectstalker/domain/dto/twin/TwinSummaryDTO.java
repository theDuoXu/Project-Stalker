package projectstalker.domain.dto.twin;

import com.fasterxml.jackson.annotation.JsonView;
import lombok.Builder;
import lombok.With;

import projectstalker.domain.sensors.SensorViews;

/**
 * DTO para crear/editar un Twin
 */
@Builder
@With
@JsonView(SensorViews.Internal.class)
public record TwinSummaryDTO(
        String id,
        String name,
        String description,
        String createdAt,
        float totalLengthKm,
        int cellCount
) {}