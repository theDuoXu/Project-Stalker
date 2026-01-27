package projectstalker.compute.api;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;
import projectstalker.compute.entity.AlertEntity;
import projectstalker.compute.repository.AlertRepository;

import java.util.List;

@RestController
@RequestMapping("/api/alerts")
@RequiredArgsConstructor
@Tag(name = "Sensores y Datos", description = "Alertas generadas por el sistema")
public class AlertController {

    private final AlertRepository alertRepository;

    @GetMapping
    @Operation(summary = "Obtener alertas paginadas por fecha")
    public projectstalker.domain.dto.PaginatedResponse<projectstalker.domain.dto.alert.AlertDTO> getAlerts(
            @RequestParam(required = false) @org.springframework.format.annotation.DateTimeFormat(iso = org.springframework.format.annotation.DateTimeFormat.ISO.DATE_TIME) java.time.LocalDateTime start,
            @RequestParam(required = false) @org.springframework.format.annotation.DateTimeFormat(iso = org.springframework.format.annotation.DateTimeFormat.ISO.DATE_TIME) java.time.LocalDateTime end,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size) {

        if (start == null)
            start = java.time.LocalDateTime.now().minusMonths(1);
        if (end == null)
            end = java.time.LocalDateTime.now();

        System.out.println("AlertController: Fetching alerts between " + start + " and " + end + " page=" + page);

        var entities = alertRepository.findByTimestampBetween(start, end,
                org.springframework.data.domain.PageRequest.of(page, size,
                        org.springframework.data.domain.Sort.by("timestamp").descending()));

        System.out.println("AlertController: Found " + entities.getTotalElements() + " alerts.");

        var dtoPage = entities.map(entity -> new projectstalker.domain.dto.alert.AlertDTO(
                entity.getId(),
                entity.getSensorId(),
                entity.getSensorId(), // Default station name
                projectstalker.domain.dto.alert.AlertSeverity.valueOf(entity.getSeverity().name()),
                projectstalker.domain.dto.alert.AlertStatus.valueOf(entity.getStatus().name()),
                entity.getMessage(),
                entity.getTimestamp(),
                0.0, // Default value
                entity.getMetric(),
                entity.getReport() != null ? entity.getReport().getId() : null));

        return new projectstalker.domain.dto.PaginatedResponse<>(
                dtoPage.getContent(),
                dtoPage.getNumber(),
                dtoPage.getSize(),
                dtoPage.getTotalElements(),
                dtoPage.getTotalPages(),
                dtoPage.isLast(),
                dtoPage.isFirst(),
                dtoPage.isEmpty());
    }

    @GetMapping("/active")
    @Operation(summary = "Obtener alertas activas (NEW y ACKNOWLEDGED)")
    public List<AlertEntity> getActiveAlerts() {
        return alertRepository.findByStatusIn(List.of(
                AlertEntity.AlertStatus.NEW,
                AlertEntity.AlertStatus.ACKNOWLEDGED,
                AlertEntity.AlertStatus.ACTIVE // Just in case
        ));
    }

    @PostMapping("/{id}/ack")
    @Operation(summary = "Reconocer (Acknowledge) una alerta")
    @org.springframework.transaction.annotation.Transactional
    public AlertEntity acknowledgeAlert(@PathVariable String id) {
        AlertEntity alert = alertRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Alert not found"));
        alert.setStatus(AlertEntity.AlertStatus.ACKNOWLEDGED);
        return alertRepository.save(alert);
    }

    @PostMapping("/{id}/resolve")
    @Operation(summary = "Resolver una alerta")
    @org.springframework.transaction.annotation.Transactional
    public AlertEntity resolveAlert(@PathVariable String id) {
        AlertEntity alert = alertRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Alert not found"));
        alert.setStatus(AlertEntity.AlertStatus.RESOLVED);
        return alertRepository.save(alert);
    }
}
