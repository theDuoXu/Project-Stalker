package projectstalker.compute.api;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import projectstalker.compute.service.ReportService;

import java.util.Map;
import java.util.UUID;

@RestController
@RequestMapping("/api/reports")
@RequiredArgsConstructor
@Tag(name = "Reporting", description = "Generación de informes PDF/CSV")
public class ReportController {

    private final ReportService reportService;

    public ResponseEntity<Map<String, String>> generateReport(@RequestBody Map<String, Object> criteria) {
        String jobId = reportService.queueReportGeneration(criteria);
        return ResponseEntity.ok(Map.of("jobId", jobId, "status", "QUEUED"));
    }

    @PostMapping("/create")
    @Operation(summary = "Crear reporte y resolver alertas asociadas")
    public ResponseEntity<projectstalker.compute.entity.ReportEntity> createReport(
            @RequestBody projectstalker.domain.dto.report.CreateReportRequest request) {
        return ResponseEntity.ok(reportService.createReport(request));
    }

    @GetMapping("/{id}")
    @Operation(summary = "Obtener detalles de un reporte")
    public ResponseEntity<projectstalker.compute.entity.ReportEntity> getReport(@PathVariable String id) {
        return ResponseEntity.ok(reportService.getReport(id));
    }

    @GetMapping("/jobs/{id}")
    @Operation(summary = "Consultar estado del trabajo", description = "Devuelve el estado (PENDING, COMPLETED, FAILED) y la URL de descarga si está listo.")
    public ResponseEntity<Map<String, Object>> getJobStatus(@PathVariable String id) {
        return ResponseEntity.ok(reportService.getJobStatus(id));
    }
}
