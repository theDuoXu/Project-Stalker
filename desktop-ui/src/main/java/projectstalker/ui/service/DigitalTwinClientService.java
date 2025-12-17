package projectstalker.ui.service;

import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatusCode;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import projectstalker.config.ApiRoutes;
import projectstalker.config.RiverConfig;
import projectstalker.domain.dto.twin.FlowPreviewRequest;
import projectstalker.domain.dto.twin.TwinCreateRequest;
import projectstalker.domain.dto.twin.TwinDetailDTO;
import projectstalker.domain.dto.twin.TwinSummaryDTO;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

@Slf4j
@Service
public class DigitalTwinClientService {

    private final WebClient apiClient;

    public DigitalTwinClientService(WebClient apiClient) {
        this.apiClient = apiClient;
    }

    // =========================================================================
    // 1. LECTURA (READ)
    // =========================================================================

    public Flux<TwinSummaryDTO> getAllTwins() {
        log.debug("Solicitando lista de proyectos a: {}", ApiRoutes.TWINS);
        return apiClient.get()
                .uri(ApiRoutes.TWINS)
                .retrieve()
                .bodyToFlux(TwinSummaryDTO.class)
                .doOnError(e -> log.error("Error recuperando proyectos: {}", e.getMessage()))
                .onErrorResume(e -> Flux.empty());
    }

    public Mono<TwinDetailDTO> getTwinDetails(String id) {
        log.debug("Solicitando detalle del proyecto ID: {}", id);
        return apiClient.get()
                .uri(ApiRoutes.TWINS + "/" + id)
                .retrieve()
                .bodyToMono(String.class)
                .map(jsonString -> {
                    log.info("JSON RAW RECIBIDO: {}", jsonString);
                    try {
                        // Deserializamos manualmente para este test
                        return new com.fasterxml.jackson.databind.ObjectMapper()
                                .findAndRegisterModules() // Importante para Records/JavaTime
                                .readValue(jsonString, TwinDetailDTO.class);
                    } catch (Exception e) {
                        throw new RuntimeException("Error manual de deserialización", e);
                    }
                })
                // ----------------------------------------------------
                .doOnError(e -> log.error("Error recuperando detalle Twin[{}]: {}", id, e.getMessage()));
    }

    // =========================================================================
    // 2. ESCRITURA (CREATE / UPDATE / DELETE)
    // =========================================================================

    public Mono<TwinSummaryDTO> createTwin(TwinCreateRequest request) {
        log.info("Enviando petición de creación de Twin: {}", request.name());
        return apiClient.post()
                .uri(ApiRoutes.TWINS)
                .bodyValue(request)
                .retrieve()
                .bodyToMono(TwinSummaryDTO.class)
                .doOnSuccess(dto -> log.info("Twin creado exitosamente con ID: {}", dto.id()))
                .doOnError(e -> log.error("Error creando Twin: {}", e.getMessage()));
    }

    public Mono<TwinDetailDTO> updateTwin(String id, TwinCreateRequest request) {
        log.info("Enviando petición de actualización para Twin: {}", id);
        return apiClient.put()
                .uri(ApiRoutes.TWINS + "/" + id)
                .bodyValue(request)
                .retrieve()
                .bodyToMono(TwinDetailDTO.class)
                .doOnError(e -> log.error("Error actualizando Twin[{}]: {}", id, e.getMessage()));
    }

    public Mono<Void> deleteTwin(String id) {
        log.warn("Solicitando eliminación del Twin: {}", id);
        return apiClient.delete()
                .uri(ApiRoutes.TWINS + "/" + id)
                .retrieve()
                .bodyToMono(Void.class)
                .doOnSuccess(v -> log.info("Twin eliminado correctamente: {}", id))
                .doOnError(e -> log.error("Error eliminando Twin[{}]: {}", id, e.getMessage()));
    }

    // =========================================================================
    // 3. SIMULACIÓN & PREVIEW (STATELESS)
    // =========================================================================

    public Mono<float[]> previewFlow(FlowPreviewRequest request) {
        String uri = ApiRoutes.TWINS + "/preview/flow";
        log.debug("Solicitando preview de flujo (Seed: {}, Duración: {}s)", request.seed(), request.durationSeconds());

        return apiClient.post()
                .uri(uri)
                .bodyValue(request)
                .retrieve()
                .bodyToMono(float[].class)
                .doOnError(e -> log.error("Error en preview de flujo: {}", e.getMessage()));
    }

    public Mono<float[]> previewTemperature(RiverConfig config, double timeOfDaySeconds) {
        String uri = ApiRoutes.TWINS + "/preview/temperature";
        log.debug("Solicitando preview de temperatura para t={}", timeOfDaySeconds);

        return apiClient.post()
                .uri(uriBuilder -> uriBuilder
                        .path(uri)
                        .queryParam("timeOfDaySeconds", timeOfDaySeconds)
                        .build())
                .bodyValue(config)
                .retrieve()
                .bodyToMono(float[].class)
                .doOnError(e -> log.error("Error en preview de temperatura: {}", e.getMessage()));
    }
}