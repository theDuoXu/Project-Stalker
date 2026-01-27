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
    @Operation(summary = "Obtener todas las alertas")
    public List<AlertEntity> getAllAlerts() {
        return alertRepository.findAll();
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
