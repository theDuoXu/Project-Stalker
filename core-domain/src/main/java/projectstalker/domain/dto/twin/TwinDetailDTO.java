package projectstalker.domain.dto.twin;

import com.fasterxml.jackson.annotation.JsonView;
import lombok.Builder;
import lombok.With;

import projectstalker.config.RiverConfig;
import projectstalker.domain.event.GeologicalEvent;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.sensors.SensorViews;

import java.util.List;

/**
 * DTO para crear/editar un Twin
 */
@Builder
@With
@JsonView(SensorViews.Internal.class)
public record TwinDetailDTO(
        String id,
        String name,
        String description,
        RiverConfig config,
        List<GeologicalEvent> events
) {}