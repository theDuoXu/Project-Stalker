package projectstalker.ui.service;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import projectstalker.domain.dto.alert.AlertDTO;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import projectstalker.domain.dto.report.ReportDTO;

@Slf4j
@Service
@lombok.RequiredArgsConstructor
public class AlertClientService {

        private final org.springframework.web.reactive.function.client.WebClient apiClient;

        public Flux<AlertDTO> getActiveAlerts() {
                return apiClient.get()
                                .uri("/api/alerts/active")
                                .retrieve()
                                .bodyToFlux(AlertDTO.class);
        }

        public reactor.core.publisher.Mono<projectstalker.ui.model.RestPage<AlertDTO>> getAlerts(int page, int size,
                        java.time.LocalDateTime start, java.time.LocalDateTime end, java.util.List<String> status) {
                return apiClient.get()
                                .uri(uriBuilder -> {
                                        var b = uriBuilder.path("/api/alerts")
                                                        .queryParam("page", page)
                                                        .queryParam("size", size)
                                                        .queryParam("start", start)
                                                        .queryParam("end", end);
                                        if (status != null && !status.isEmpty()) {
                                                b.queryParam("status", status);
                                        }
                                        return b.build();
                                })
                                .retrieve()
                                .bodyToMono(
                                                new org.springframework.core.ParameterizedTypeReference<projectstalker.ui.model.RestPage<AlertDTO>>() {
                                                });
        }

        public Mono<Void> acknowledgeAlert(String alertId) {
                return apiClient.post()
                                .uri("/api/alerts/{id}/ack", alertId)
                                .retrieve()
                                .bodyToMono(Void.class);
        }

        public Mono<Void> resolveAlert(String alertId) {
                return apiClient.post()
                                .uri("/api/alerts/{id}/resolve", alertId)
                                .retrieve()
                                .bodyToMono(Void.class);
        }

        public Mono<Object> createReport(projectstalker.domain.dto.report.CreateReportRequest request) {
                return apiClient.post()
                                .uri("/api/reports/create")
                                .bodyValue(request)
                                .retrieve()
                                .bodyToMono(Object.class);
        }

        public Mono<ReportDTO> getReport(String id) {
                return apiClient.get()
                                .uri("/api/reports/{id}", id)
                                .retrieve()
                                .bodyToMono(ReportDTO.class);
        }
}
