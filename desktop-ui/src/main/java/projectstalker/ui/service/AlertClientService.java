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

    private final org.springframework.web.reactive.function.client.WebClient webClient;

    public AlertClientService(org.springframework.web.reactive.function.client.WebClient.Builder webClientBuilder) {
        this.webClient = webClientBuilder.baseUrl("http://localhost:8080").build();
    }

    public Flux<AlertDTO> getActiveAlerts() {
        return webClient.get()
                .uri("/api/alerts/active")
                .retrieve()
                .bodyToFlux(AlertDTO.class);
    }

    public Mono<Void> acknowledgeAlert(String alertId) {
        return webClient.post()
                .uri("/api/alerts/{id}/ack", alertId)
                .retrieve()
                .bodyToMono(Void.class);
    }
}
