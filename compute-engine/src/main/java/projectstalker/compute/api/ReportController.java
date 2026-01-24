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
@RequestMapping("/reports")
@RequiredArgsConstructor
@Tag(name = "Reporting", description = "Generación de informes PDF/CSV")
public class ReportController {

    private final ReportService reportService;

    @PostMapping("/generate")
    @Operation(summary = "Solicitar generación de informe", description = "Inicia un trabajo asíncrono y devuelve el Job ID.")
    public ResponseEntity<Map<String, String>> generateReport(@RequestBody Map<String, Object> criteria) {
        String jobId = reportService.queueReportGeneration(criteria);
        return ResponseEntity.ok(Map.of("jobId", jobId, "status", "QUEUED"));
    }

    @GetMapping("/jobs/{id}")
    @Operation(summary = "Consultar estado del trabajo", description = "Devuelve el estado (PENDING, COMPLETED, FAILED) y la URL de descarga si está listo.")
    public ResponseEntity<Map<String, Object>> getJobStatus(@PathVariable String id) {
        return ResponseEntity.ok(reportService.getJobStatus(id));
    }
}
