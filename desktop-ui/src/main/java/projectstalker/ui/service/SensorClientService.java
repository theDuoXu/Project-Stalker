package projectstalker.ui.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import projectstalker.config.ApiRoutes;
import projectstalker.domain.dto.sensor.SensorCreationDTO;
import projectstalker.domain.dto.sensor.SensorHealthResponseDTO;
import projectstalker.domain.dto.sensor.SensorReadingDTO;
import projectstalker.domain.dto.sensor.SensorResponseDTO;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

@Slf4j
@Service
@RequiredArgsConstructor
public class SensorClientService {

    private final WebClient apiClient;

    // =========================================================================
    // 1. ESCRITURA (Gestión de Inventario)
    // =========================================================================

    /**
     * Crea un nuevo sensor en el sistema (Virtual o Real).
     * Requiere rol ANALYST o ADMIN.
     */
    public Mono<SensorResponseDTO> createSensor(SensorCreationDTO request) {
        log.info("Enviando petición de creación de Sensor: {} (Tipo: {})", request.name(), request.strategyType());

        // Asumimos que el POST va a la raíz de /api/sensors
        // Si tu backend usa otra ruta (ej: /api/sensors/create), ajústalo aquí.
        return apiClient.post()
                .uri(ApiRoutes.SENSORS)
                .bodyValue(request)
                .retrieve()
                .bodyToMono(SensorResponseDTO.class)
                .doOnSuccess(dto -> log.info("Sensor creado exitosamente. ID/Station: {}", dto.stationId()))
                .doOnError(e -> log.error("Error creando Sensor: {}", e.getMessage()));
    }

    // =========================================================================
    // 2. LECTURA (Monitorización)
    // =========================================================================

    public Mono<SensorResponseDTO> getHistory(String stationId, String parameter) {
        log.debug("Solicitando histórico de Sensor[{}] Param: {}", stationId, parameter);

        return apiClient.get()
                .uri(uriBuilder -> uriBuilder
                        .path(ApiRoutes.SENSORS + "/{stationId}/history")
                        .queryParam("parameter", parameter)
                        .build(stationId))
                .retrieve()
                .bodyToMono(SensorResponseDTO.class)
                .doOnError(e -> log.error("Error recuperando histórico [{}]: {}", stationId, e.getMessage()));
    }

    public Flux<SensorReadingDTO> getRealtime(String stationId, String parameter) {
        // Nota: El backend devuelve List<SensorReadingDTO>, WebClient lo convierte a Flux automáticamente
        // si usamos bodyToFlux, iterando sobre la lista JSON.

        return apiClient.get()
                .uri(uriBuilder -> uriBuilder
                        .path(ApiRoutes.SENSORS + "/{stationId}/realtime")
                        .queryParam("parameter", parameter)
                        .build(stationId))
                .retrieve()
                .bodyToFlux(SensorReadingDTO.class)
                .doOnError(e -> log.error("Error en lectura Realtime [{}]: {}", stationId, e.getMessage()))
                .onErrorResume(e -> Flux.empty()); // Resiliencia: Si falla un sensor, no mata la UI
    }

    public Mono<SensorHealthResponseDTO> getHealthStatus(String stationId) {
        return apiClient.get()
                .uri(uriBuilder -> uriBuilder
                        .path(ApiRoutes.SENSORS + "/{stationId}/status")
                        .queryParam("parameter", "ALL")
                        .build(stationId))
                .retrieve()
                .bodyToMono(SensorHealthResponseDTO.class)
                .doOnError(e -> log.warn("No se pudo obtener estado de salud de [{}]: {}", stationId, e.getMessage()))
                // Si falla el health check, devolvemos un objeto vacío o null para que la UI muestre "Desconocido"
                .onErrorResume(e -> Mono.empty());
    }

    // =========================================================================
    // 3. EXPORTACIÓN
    // =========================================================================

    public Mono<SensorResponseDTO> exportData(String stationId, String parameter, LocalDateTime from, LocalDateTime to) {
        log.info("Solicitando exportación de datos [{}] desde {} hasta {}", stationId, from, to);

        return apiClient.get()
                .uri(uriBuilder -> uriBuilder
                        .path(ApiRoutes.SENSORS + "/export/{stationId}")
                        .queryParam("parameter", parameter)
                        .queryParam("from", from.format(DateTimeFormatter.ISO_DATE_TIME))
                        .queryParam("to", to.format(DateTimeFormatter.ISO_DATE_TIME))
                        .build(stationId))
                .retrieve()
                .bodyToMono(SensorResponseDTO.class)
                .doOnError(e -> log.error("Fallo en exportación: {}", e.getMessage()));
    }
}