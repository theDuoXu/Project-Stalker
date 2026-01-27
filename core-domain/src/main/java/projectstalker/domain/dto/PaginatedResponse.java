package projectstalker.domain.dto;

import com.fasterxml.jackson.annotation.JsonView;
import projectstalker.domain.sensors.SensorViews;
import java.util.List;

@JsonView(SensorViews.Public.class)
public record PaginatedResponse<T>(
        List<T> content,
        int number,
        int size,
        long totalElements,
        int totalPages,
        boolean last,
        boolean first,
        boolean empty) {
}
