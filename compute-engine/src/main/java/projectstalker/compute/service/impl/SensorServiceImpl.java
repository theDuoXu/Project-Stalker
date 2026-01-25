package projectstalker.compute.service.impl;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import projectstalker.compute.entity.SensorEntity;
import projectstalker.compute.repository.DigitalTwinRepository;
import projectstalker.compute.repository.SensorRepository;
import projectstalker.compute.service.SensorService;
import projectstalker.domain.dto.sensor.*;
import projectstalker.domain.exception.InvalidExportRequestException;
import projectstalker.domain.exception.SensorBusinessException;
import projectstalker.domain.sensors.*;

import java.time.Duration;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;

@Slf4j
@Service
public class SensorServiceImpl implements SensorService {
    private final DigitalTwinRepository twinRepository;
    private final SensorRepository sensorRepository;
    private final org.springframework.web.reactive.function.client.WebClient webClient;
    private final com.fasterxml.jackson.databind.ObjectMapper objectMapper = new com.fasterxml.jackson.databind.ObjectMapper();
    private final java.util.Map<String, java.util.Map.Entry<LocalDateTime, List<SensorReadingDTO>>> webhookCache = new java.util.concurrent.ConcurrentHashMap<>();
    private final java.util.Map<String, String> saicaMasterList = new java.util.concurrent.ConcurrentHashMap<>();

    public SensorServiceImpl(DigitalTwinRepository twinRepository, SensorRepository sensorRepository,
            org.springframework.web.reactive.function.client.WebClient.Builder webClientBuilder) {
        this.twinRepository = twinRepository;
        this.sensorRepository = sensorRepository;
        this.webClient = webClientBuilder.build();
        loadSaicaMasterList();
    }

    private void loadSaicaMasterList() {
        try {
            java.io.InputStream is = getClass().getResourceAsStream("/saica-stations.json");
            if (is == null) {
                log.warn("SAICA Master List not found in classpath (/saica-stations.json)");
                return;
            }
            com.fasterxml.jackson.databind.JsonNode root = objectMapper.readTree(is);
            if (root.isArray()) {
                for (com.fasterxml.jackson.databind.JsonNode node : root) {
                    if (node.has("codigo") && node.has("url_detalles")) {
                        String code = node.get("codigo").asText();
                        String url = node.get("url_detalles").asText();
                        saicaMasterList.put(code, url);
                    }
                }
                log.info("Loaded {} SAICA stations from master list.", saicaMasterList.size());
            }
        } catch (Exception e) {
            log.error("Failed to load SAICA Master List", e);
        }
    }

    @Override
    public SensorResponseDTO getHistory(String stationId, String parameter) {
        String paramNormalized = parameter.toUpperCase();
        SensorType sensorType = SensorType.fromString(parameter);

        if (sensorType == SensorType.UNKNOWN) {
            throw new IllegalArgumentException(
                    "Parameter: " + paramNormalized + " does not match any known sensor type");
        }

        // CHECK IF VIRTUAL OR MANUAL SENSOR
        var entityOpt = sensorRepository.findById(stationId);
        if (entityOpt.isPresent()) {
            String sType = entityOpt.get().getStrategyType();
            // Handle all non-REAL types (VIRTUAL, MANUAL, etc)
            if ("VIRTUAL".equalsIgnoreCase(sType) || "MANUAL".equalsIgnoreCase(sType)) {
                List<SensorReadingDTO> generated = generateVirtualReadings(entityOpt.get());
                return SensorResponseDTO.builder()
                        .stationId(stationId)
                        .name(entityOpt.get().getName()) // Use real name
                        .signalType(sType)
                        .unit(sensorType.getUnit())
                        .values(generated)
                        .configuration(entityOpt.get().getConfiguration())
                        .typeCode(entityOpt.get().getType().getCode())
                        .build();
            }
        }

        List<SensorReadingDTO> readings = sensorRepository.findReadings(stationId, paramNormalized);

        return SensorResponseDTO.builder()
                .stationId(stationId)
                .name(sensorType.getCode())
                .signalType(sensorType.getSignalType())
                .unit(sensorType.getUnit())
                .values(readings)
                .build();
    }

    private List<SensorReadingDTO> generatePhysicsProfile(SensorEntity entity, double base, double amp, double freq) {
        // Use Entity ID hash as seed for consistency
        int seed = entity.getId().hashCode();
        projectstalker.physics.model.RandomFlowProfileGenerator generator = new projectstalker.physics.model.RandomFlowProfileGenerator(
                seed, base, amp, (float) freq);

        List<SensorReadingDTO> readings = new java.util.ArrayList<>();
        LocalDateTime now = LocalDateTime.now();
        // Generate 60 mins of data
        // Generator works with seconds. 60 mins = 3600 sec.
        // We want data relative to NOW.
        // Let's assume t=0 is NOW-60mins.

        for (int i = 60; i >= 0; i--) {
            // Time relative to start of generation window
            double t = (60 - i) * 60.0; // Minutes to Seconds
            double val = generator.getDischargeAt(t);

            readings.add(SensorReadingDTO.builder()
                    .stationId(entity.getId())
                    .tag(entity.getName())
                    .timestamp(now.minusMinutes(i).toString())
                    .value(val)
                    .formattedValue(String.format("%.2f", val))
                    .build());
        }
        return readings;
    }

    private List<SensorReadingDTO> generateVirtualReadings(SensorEntity entity) {
        java.util.Map<String, Object> config = entity.getConfiguration();
        String algo = (String) config.getOrDefault("strategy", "VIRTUAL_SINE");

        if ("VIRTUAL_PHYSICS_FLOW".equals(algo)) {
            double base = config.containsKey("baseDischarge") ? ((Number) config.get("baseDischarge")).doubleValue()
                    : 25.0;
            double amp = config.containsKey("noiseAmplitude") ? ((Number) config.get("noiseAmplitude")).doubleValue()
                    : 5.0;
            double freq = config.containsKey("frequency") ? ((Number) config.get("frequency")).doubleValue() : 0.02;
            return generatePhysicsProfile(entity, base, amp, freq);
        }

        if ("MANUAL_STATIC".equals(algo) || "MANUAL".equalsIgnoreCase(entity.getStrategyType())) {
            double val = config.containsKey("value") ? ((Number) config.get("value")).doubleValue() : 0.0;
            return generateConstantProfile(entity, val); // Need to define this too if missing
        }

        if ("VIRTUAL_RANDOM".equals(algo)) {
            double min = config.containsKey("min") ? ((Number) config.get("min")).doubleValue() : 0.0;
            double max = config.containsKey("max") ? ((Number) config.get("max")).doubleValue() : 10.0;
            return generateRandomProfile(entity, min, max); // Need to define this too if missing
        }

        double amplitude = config.containsKey("amplitude") ? ((Number) config.get("amplitude")).doubleValue() : 1.0;
        double frequency = config.containsKey("frequency") ? ((Number) config.get("frequency")).doubleValue() : 0.1;
        double offset = config.containsKey("offset") ? ((Number) config.get("offset")).doubleValue() : 2.0;

        // Generate 100 points for the last hour? Or 24h?
        // Let's generate last 60 minutes
        List<SensorReadingDTO> readings = new java.util.ArrayList<>();
        LocalDateTime now = LocalDateTime.now();

        for (int i = 60; i >= 0; i--) {
            LocalDateTime time = now.minusMinutes(i);
            // Simple Sine Wave: y = A * sin(f * t) + offset
            // We use 'i' as time step
            double t = (60 - i);
            double value = amplitude * Math.sin(frequency * t) + offset;

            readings.add(SensorReadingDTO.builder()
                    .stationId(entity.getId())
                    .tag(entity.getName())
                    .timestamp(time.toString()) // ISO format
                    .value(value)
                    .formattedValue(String.format("%.2f", value))
                    .build());
        }
        return readings;
    }

    private List<SensorReadingDTO> generateConstantProfile(SensorEntity entity, double val) {
        List<SensorReadingDTO> readings = new java.util.ArrayList<>();
        LocalDateTime now = LocalDateTime.now();
        for (int i = 60; i >= 0; i--) {
            readings.add(SensorReadingDTO.builder()
                    .stationId(entity.getId())
                    .tag(entity.getName())
                    .timestamp(now.minusMinutes(i).toString())
                    .value(val)
                    .formattedValue(String.format("%.2f", val))
                    .build());
        }
        return readings;
    }

    private List<SensorReadingDTO> generateRandomProfile(SensorEntity entity, double min, double max) {
        List<SensorReadingDTO> readings = new java.util.ArrayList<>();
        LocalDateTime now = LocalDateTime.now();
        java.util.Random rnd = new java.util.Random();
        for (int i = 60; i >= 0; i--) {
            double val = min + (max - min) * rnd.nextDouble();
            readings.add(SensorReadingDTO.builder()
                    .stationId(entity.getId())
                    .tag(entity.getName())
                    .timestamp(now.minusMinutes(i).toString())
                    .value(val)
                    .formattedValue(String.format("%.2f", val))
                    .build());
        }
        return readings;
    }

    @Override
    public SensorHealthResponseDTO getHealth(String stationId, String parameter) {
        String paramUpper = parameter.toUpperCase();
        List<SensorHealthDTO> healthData;

        if ("ALL".equals(paramUpper)) {
            healthData = sensorRepository.findHealthStatus(stationId);
        } else {
            SensorType type = SensorType.fromString(paramUpper);
            if (type == SensorType.UNKNOWN) {
                healthData = List.of();
            } else {
                healthData = sensorRepository.findHealthStatusByType(stationId, type);
            }
        }

        boolean isAllOk = healthData.stream()
                .allMatch(h -> (h.batteryPercentage() > 20) &&
                        (Duration.between(h.lastChecked(), LocalDateTime.now()).toHours() < 24));

        return SensorHealthResponseDTO.builder()
                .stationId(stationId)
                .isAllOk(isAllOk)
                .values(healthData)
                .build();
    }

    @Override
    public List<SensorReadingDTO> getRealtime(String stationId, String parameter) {
        String paramUpper = parameter.toUpperCase();
        log.info("getRealtime called for station={} param={}", stationId, paramUpper);

        var entityOpt = sensorRepository.findById(stationId);
        if (entityOpt.isPresent()) {
            SensorEntity sensor = entityOpt.get();
            String sType = sensor.getStrategyType();
            log.info("Station {} found. Strategy: {}", stationId, sType);

            // 1. Virtual / Manual Stategies
            if ("VIRTUAL".equalsIgnoreCase(sType) || "MANUAL".equalsIgnoreCase(sType)) {
                List<SensorReadingDTO> history = generateVirtualReadings(sensor);
                if (!history.isEmpty()) {
                    return List.of(history.get(history.size() - 1));
                }
            }

            // 2. REAL WEBHOOK PROXY (On-Demand with Caching)
            if ("REAL_IoT_WEBHOOK".equals(sType) || "REAL".equals(sType)) {
                java.util.Map<String, Object> config = sensor.getConfiguration();
                if (config != null && config.containsKey("url")) {
                    String url = (String) config.get("url");
                    return fetchAndParseSaica(stationId, sensor.getName(), url);
                } else {
                    log.warn("Sensor {} has REAL_IoT_WEBHOOK but NO URL in config", sensor.getName());
                }
            }
        } else {
            // NOT IN DB? Check Master List for SAICA Code (e.g. C326)
            if (saicaMasterList.containsKey(stationId)) {
                String url = saicaMasterList.get(stationId);
                log.info("Station {} found in SAICA Master List. Proxying to: {}", stationId, url);
                return fetchAndParseSaica(stationId, "SAICA-" + stationId, url);
            }

            log.warn("Station {} NOT FOUND in DB and NOT in SAICA Master List", stationId);
        }

        if ("ALL".equals(paramUpper)) {
            List<SensorReadingDTO> dbReadings = sensorRepository.findLatestReadings(stationId);
            log.info("Falling back to DB. Found {} readings", dbReadings.size());
            return dbReadings;
        }

        SensorType type = SensorType.fromString(paramUpper);
        if (type == SensorType.UNKNOWN) {
            return List.of();
        }
        return sensorRepository.findLatestReadingsByType(stationId, type);
    }

    @Override
    public SensorResponseDTO getExportData(String stationId, String parameter, LocalDateTime from, LocalDateTime to) {
        String paramUpper = parameter.toUpperCase();

        if ("ALL".equals(paramUpper)) {
            throw new SensorBusinessException("Bulk export for ALL parameters is restricted.");
        }

        SensorType type = SensorType.fromString(paramUpper);
        if (type == SensorType.UNKNOWN) {
            throw new InvalidExportRequestException("Unknown parameter: '" + parameter + "'.");
        }

        if (to == null)
            to = LocalDateTime.now();
        if (from == null)
            from = LocalDateTime.now().minusYears(100);

        List<SensorReadingDTO> readings = sensorRepository.findReadingsByDateRange(stationId, type.getCode(), from, to);

        return SensorResponseDTO.builder()
                .stationId(stationId)
                .name(type.getCode())
                .signalType(type.getSignalType())
                .unit(type.getUnit())
                .values(readings)
                .build();
    }

    @Override
    @Transactional
    public SensorResponseDTO registerSensor(SensorCreationDTO request) {
        log.info("Registrando nuevo sensor: {}", request.name());

        // 1. Validaciones
        SensorType type = SensorType.fromString(request.type());
        if (type == SensorType.UNKNOWN)
            throw new SensorBusinessException("Tipo inválido");

        // Necesitamos el ID del Gemelo
        String twinId = request.twinId();
        if (twinId == null || twinId.isBlank()) {
            throw new SensorBusinessException("El sensor debe estar asociado a un Gemelo Digital válido.");
        }

        // Configuración Segura (Null Safe)
        java.util.Map<String, Object> config = request.configuration();
        if (config == null)
            config = java.util.Collections.emptyMap();

        // Recuperar la entidad del Río (Referencia JPA)
        // Usamos getReferenceById si solo queremos enlazar sin hacer SELECT,
        // pero findById es más seguro para validar que el río existe.
        var river = twinRepository.findById(twinId)
                .orElseThrow(() -> new SensorBusinessException("El Río (Twin) especificado no existe."));

        // 2. Construir Entidad
        SensorEntity entity = SensorEntity.builder()
                .name(request.name())
                .type(type)
                .locationKm(request.locationKm())
                .strategyType(request.strategyType())
                .configuration(request.configuration()) // JSONB automático
                .twin(river) // Relación FK
                .isActive(true)
                .build();

        // 3. Guardar (El ID UUID se genera aquí si no lo seteaste)
        SensorEntity saved = sensorRepository.save(entity);

        // 4. Retornar DTO con el ID generado
        return SensorResponseDTO.builder()
                .stationId(saved.getId()) // UUID real
                .name(saved.getName())
                .signalType(saved.getStrategyType()) // Correct mapping for Frontend
                .unit(saved.getType().getUnit())
                .values(List.of())
                .build();
    }

    @Override
    public List<SensorResponseDTO> getAllByTwin(String twinId) {
        List<SensorEntity> sensors = sensorRepository.findAllByTwinId(twinId);
        return sensors.stream().map(s -> SensorResponseDTO.builder()
                .stationId(s.getId())
                .name(s.getName())
                .signalType(s.getStrategyType()) // Correct mapping for Frontend
                .unit(s.getType().getUnit())
                .values(List.of()) // Only metadata for list view
                .typeCode(s.getType().getCode())
                .configuration(s.getConfiguration())
                .build()).toList();
    }

    @Override
    @Transactional
    public SensorResponseDTO updateSensor(String stationId, SensorCreationDTO request) {
        log.info("Actualizando sensor [{}]: {}", stationId, request.name());

        SensorEntity entity = sensorRepository.findById(stationId)
                .orElseThrow(() -> new SensorBusinessException("Sensor no encontrado: " + stationId));

        // Update fields
        entity.setName(request.name());
        entity.setLocationKm(request.locationKm());

        // Update type (with validation)
        SensorType type = SensorType.fromString(request.type());
        if (type != SensorType.UNKNOWN) {
            entity.setType(type);
        }

        // Update Strategy & Config
        if (request.strategyType() != null) {
            entity.setStrategyType(request.strategyType());
        }
        if (request.configuration() != null) {
            entity.setConfiguration(request.configuration());
        }

        SensorEntity saved = sensorRepository.save(entity);

        return SensorResponseDTO.builder()
                .stationId(saved.getId())
                .name(saved.getName())
                .signalType(saved.getStrategyType())
                .unit(saved.getType().getUnit())
                .values(List.of())
                .build();
    }

    private List<SensorReadingDTO> fetchAndParseSaica(String stationId, String stationName, String url) {
        // Check Cache (15 min TTL)
        var cached = webhookCache.get(url);
        if (cached != null && java.time.Duration.between(cached.getKey(), LocalDateTime.now()).toMinutes() < 15) {
            log.debug("Returning cached data for {}", url);
            return cached.getValue();
        }

        log.info("Proxying request to Webhook: {}", url);
        try {
            // Synchronous Fetch
            String body = webClient.get().uri(url)
                    .retrieve()
                    .bodyToMono(String.class)
                    .block(Duration.ofSeconds(10));

            if (body != null) {
                List<SensorReadingDTO> parsedReadings = new java.util.ArrayList<>();
                boolean isSaica = false;

                // Try parsing as SAICA JSON
                try {
                    com.fasterxml.jackson.databind.JsonNode root = objectMapper.readTree(body);
                    if (root.has("response") && root.get("response").has("senales")) {
                        isSaica = true;
                        com.fasterxml.jackson.databind.JsonNode senales = root.get("response").get("senales");
                        DateTimeFormatter saicaFmt = DateTimeFormatter.ofPattern("dd/MM/yyyy HH:mm");

                        for (com.fasterxml.jackson.databind.JsonNode senal : senales) {
                            String tag = senal.has("nombre") ? senal.get("nombre").asText() : "UNKNOWN";
                            if (senal.has("valores")) {
                                for (com.fasterxml.jackson.databind.JsonNode valNode : senal.get("valores")) {
                                    String tStr = valNode.get("tiempo").asText();
                                    double val = valNode.get("valor").asDouble();

                                    parsedReadings.add(SensorReadingDTO.builder()
                                            .stationId(stationId)
                                            .tag(tag)
                                            .timestamp(LocalDateTime.parse(tStr, saicaFmt).toString())
                                            .value(val)
                                            .formattedValue(String.format("%.2f", val))
                                            .build());
                                }
                            }
                        }
                    }
                } catch (Exception e) {
                    if (isSaica)
                        log.error("Error parsing SAICA details", e);
                }

                if (!parsedReadings.isEmpty()) {
                    // Cache result
                    log.info("Parsed {} readings from SAICA webhook.", parsedReadings.size());
                    webhookCache.put(url, java.util.Map.entry(LocalDateTime.now(), parsedReadings));
                    return parsedReadings;
                }

                // Fallback: Simple Numeric
                try {
                    double val = Double.parseDouble(body.trim());
                    SensorReadingDTO reading = SensorReadingDTO.builder()
                            .stationId(stationId)
                            .tag("value")
                            .timestamp(LocalDateTime.now().toString())
                            .value(val)
                            .formattedValue(String.format("%.2f", val))
                            .build();
                    return List.of(reading);
                } catch (NumberFormatException e) {
                    log.warn("Could not parse numeric reading from {}: {}", url,
                            body.substring(0, Math.min(body.length(), 100)));
                }
            }
        } catch (Exception e) {
            log.error("Failed to proxy realtime webhook for {}: {}", stationName, e.getMessage());
        }
        return List.of();
    }
}