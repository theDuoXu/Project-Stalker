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
@lombok.RequiredArgsConstructor
public class AlertClientService {

    private final org.springframework.web.reactive.function.client.WebClient apiClient;

    // Manual constructor removed in favor of @RequiredArgsConstructor and injected
    // client

    public Flux<AlertDTO> getActiveAlerts() {
        return apiClient.get()
                .uri("/api/alerts/active")
                .retrieve()
                .bodyToFlux(AlertDTO.class);
    }

    public Flux<AlertDTO> getAllAlerts() {
        return apiClient.get()
                .uri("/api/alerts")
                .retrieve()
                .bodyToFlux(AlertDTO.class);
    }

    public Mono<AlertDTO> acknowledgeAlert(String alertId) {
        return apiClient.post()
                .uri("/api/alerts/{id}/ack", alertId)
                .retrieve()
                .bodyToMono(AlertDTO.class);
    }

    public Mono<AlertDTO> resolveAlert(String alertId) {
        return apiClient.post()
                .uri("/api/alerts/{id}/resolve", alertId)
                .retrieve()
                .bodyToMono(AlertDTO.class);
    }

    public Mono<Object> createReport(projectstalker.domain.dto.report.CreateReportRequest request) {
        return apiClient.post()
                .uri("/api/reports/create")
                .bodyValue(request)
                .retrieve()
                .bodyToMono(Object.class);
    }
}
