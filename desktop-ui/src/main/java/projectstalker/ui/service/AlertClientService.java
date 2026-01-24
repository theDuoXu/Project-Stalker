package projectstalker.ui.service;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import projectstalker.domain.dto.alert.AlertDTO;
import projectstalker.domain.dto.alert.AlertSeverity;
import projectstalker.domain.dto.alert.AlertStatus;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.time.LocalDateTime;
import java.util.List;
import java.util.UUID;

@Slf4j
@Service
public class AlertClientService {

    // Simular base de datos local
    private final List<AlertDTO> MOCK_ALERTS = List.of(
            new AlertDTO(UUID.randomUUID().toString(), "C302", "SAICA ARANJUEZ", AlertSeverity.CRITICAL,
                    AlertStatus.ACTIVE, "Nivel de Amonio crítico (> 2.5 mg/L)", LocalDateTime.now().minusMinutes(5),
                    2.8, "NH4"),
            new AlertDTO(UUID.randomUUID().toString(), "C316", "SAICA TRILLO", AlertSeverity.WARNING,
                    AlertStatus.ACTIVE, "Temperatura elevada", LocalDateTime.now().minusHours(1), 28.5, "TEMP"),
            new AlertDTO(UUID.randomUUID().toString(), "C322", "SAICA CARCABOSO", AlertSeverity.INFO,
                    AlertStatus.RESOLVED, "Pérdida de conexión momentánea", LocalDateTime.now().minusDays(1), 0.0,
                    "CONN"));

    public Flux<AlertDTO> getActiveAlerts() {
        // En el futuro: WebClient.get().uri("/api/alerts")...
        log.info("Obteniendo alertas (MOCK)...");
        return Flux.fromIterable(MOCK_ALERTS);
    }

    public Mono<Void> acknowledgeAlert(String alertId) {
        log.info("Alerta confirmada: {}", alertId);
        // Simulamos latencia de red
        return Mono.empty(); // En backend real esto sería un PUT
    }
}
