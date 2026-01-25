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
                log.info("Enviando petición de creación de Sensor: {} (Tipo: {})", request.name(),
                                request.strategyType());

                return apiClient.post()
                                .uri(ApiRoutes.SENSORS)
                                .bodyValue(request)
                                .retrieve()
                                .bodyToMono(SensorResponseDTO.class)
                                .doOnSuccess(dto -> log.info("Sensor creado exitosamente. ID/Station: {}",
                                                dto.stationId()))
                                .doOnError(e -> log.error("Error creando Sensor: {}", e.getMessage()));
        }

        public Mono<SensorResponseDTO> updateSensor(String stationId, SensorCreationDTO request) {
                log.info("Enviando petición de actualización de Sensor: {}", stationId);
                return apiClient.put()
                                .uri(ApiRoutes.SENSORS + "/{id}", stationId)
                                .bodyValue(request)
                                .retrieve()
                                .bodyToMono(SensorResponseDTO.class)
                                .doOnSuccess(dto -> log.info("Sensor actualizado exitosamente."))
                                .doOnError(e -> log.error("Error actualizando Sensor: {}", e.getMessage()));
        }

        public Flux<SensorResponseDTO> getSensorsByTwin(String twinId) {
                return apiClient.get()
                                .uri(uriBuilder -> uriBuilder
                                                .path(ApiRoutes.SENSORS)
                                                .queryParam("twinId", twinId)
                                                .build())
                                .retrieve()
                                .bodyToFlux(SensorResponseDTO.class)
                                .doOnError(e -> log.error("Error fetching sensors for twin {}: {}", twinId,
                                                e.getMessage()));
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
                                .doOnError(e -> log.error("Error recuperando histórico [{}]: {}", stationId,
                                                e.getMessage()));
        }

        public Flux<SensorReadingDTO> getRealtime(String stationId, String parameter) {
                // Nota: El backend devuelve List<SensorReadingDTO>, WebClient lo convierte a
                // Flux automáticamente
                // si usamos bodyToFlux, iterando sobre la lista JSON.

                return apiClient.get()
                                .uri(uriBuilder -> uriBuilder
                                                .path(ApiRoutes.SENSORS + "/{stationId}/realtime")
                                                .queryParam("parameter", parameter)
                                                .build(stationId))
                                .retrieve()
                                .bodyToFlux(SensorReadingDTO.class)
                                .doOnError(e -> log.error("Error en lectura Realtime [{}]: {}", stationId,
                                                e.getMessage()))
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
                                .doOnError(e -> log.warn("No se pudo obtener estado de salud de [{}]: {}", stationId,
                                                e.getMessage()))
                                // Si falla el health check, devolvemos un objeto vacío o null para que la UI
                                // muestre "Desconocido"
                                .onErrorResume(e -> Mono.empty());
        }

        // =========================================================================
        // 3. EXPORTACIÓN
        // =========================================================================

        public Mono<SensorResponseDTO> exportData(String stationId, String parameter, LocalDateTime from,
                        LocalDateTime to) {
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
        // =========================================================================
        // 4. METADATA & DISCOVERY (Mocked from Local JSON)
        // =========================================================================

        public Flux<projectstalker.domain.dto.sensor.SensorResponseDTO> getAllAvailableSensors() {
                // En un escenario real, esto llamaría a GET /api/sensors
                // Para la demo, cargamos el JSON local que ya tenemos
                return Flux.create(sink -> {
                        try {
                                // Usamos ObjectMapper para parsear el JSON de recursos
                                com.fasterxml.jackson.databind.ObjectMapper mapper = new com.fasterxml.jackson.databind.ObjectMapper();

                                // Leemos el fichero de recursos
                                org.springframework.core.io.ClassPathResource resource = new org.springframework.core.io.ClassPathResource(
                                                "maps/saica_stations_maestro_coords.json");

                                // El JSON es una lista de objetos con fields: codigo, nombre, subcuenca...
                                com.fasterxml.jackson.databind.JsonNode root = mapper
                                                .readTree(resource.getInputStream());

                                if (root.isArray()) {
                                        for (com.fasterxml.jackson.databind.JsonNode node : root) {
                                                // Mapeamos lo que tenemos al DTO estándar
                                                String code = node.path("codigo").asText();
                                                String name = node.path("nombre").asText();
                                                // String river = node.path("subcuenca").asText();

                                                // --- MOCK FLOW PROFILE GENERATION ---
                                                // Use station code hashCode as seed for stability
                                                int seed = code.hashCode();
                                                // Random base flow between 5 and 50 m3/s
                                                double baseFlow = 5.0 + (Math.abs(seed) % 45);
                                                // Random amplitude
                                                double amplitude = baseFlow * 0.2;

                                                projectstalker.physics.model.RandomFlowProfileGenerator generator = new projectstalker.physics.model.RandomFlowProfileGenerator(
                                                                seed, baseFlow, amplitude, 0.01f);

                                                // Generar últimas 24h (96 puntos de 15 min)
                                                java.util.List<SensorReadingDTO> readings = new java.util.ArrayList<>();
                                                LocalDateTime now = LocalDateTime.now();
                                                LocalDateTime startTime = now.minusHours(24);

                                                // Generate profile for 24h (86400 seconds) with 900s step
                                                float[] profile = generator.generateProfile(86400, 900);

                                                for (int i = 0; i < profile.length; i++) {
                                                        LocalDateTime t = startTime.plusMinutes(i * 15L);
                                                        double val = profile[i];
                                                        readings.add(new SensorReadingDTO(
                                                                        "FLOW",
                                                                        t.format(DateTimeFormatter.ISO_LOCAL_DATE_TIME),
                                                                        val,
                                                                        String.format("%.2f", val),
                                                                        code));
                                                }

                                                // Construimos un DTO "fake" con datos poblados
                                                SensorResponseDTO dto = new SensorResponseDTO(
                                                                code, // stationId
                                                                name, // name
                                                                "REAL", // signalType (usamos como estrategia/tipo)
                                                                "m³/s", // unit
                                                                readings, // values
                                                                java.util.Collections.emptyMap(),
                                                                "FLOW");
                                                sink.next(dto);
                                        }
                                }
                                sink.complete();
                        } catch (Exception e) {
                                log.error("Error cargando sensores mock", e);
                                sink.error(e);
                        }
                });
        }
}