package projectstalker.domain.dto.report;

import java.time.LocalDateTime;

public record ReportDTO(
        String id,
        String title,
        String body,
        LocalDateTime createdAt) {
}
