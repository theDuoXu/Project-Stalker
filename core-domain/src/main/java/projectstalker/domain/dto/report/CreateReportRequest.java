package projectstalker.domain.dto.report;

import java.util.List;

public record CreateReportRequest(
        String title,
        String body,
        List<String> alertIds) {
}
