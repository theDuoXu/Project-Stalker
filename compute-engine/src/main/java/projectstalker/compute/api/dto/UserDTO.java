package projectstalker.compute.api.dto;

import com.fasterxml.jackson.annotation.JsonView;
import projectstalker.domain.sensors.SensorViews;

import java.util.UUID;

@JsonView(SensorViews.Public.class)
public record UserDTO(
                UUID id,
                String username,
                String email,
                String rol) {
}
