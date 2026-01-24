package projectstalker.compute.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import projectstalker.compute.entity.SensorEntity;
import projectstalker.compute.repository.SensorRepository;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.LocalDateTime;
import java.util.*;

@Service
@Slf4j
@RequiredArgsConstructor
public class InferenceService {

    private final WebClient webClient = WebClient.create("http://inference-engine:8000"); // Container name
    private final ObjectMapper objectMapper;
    private final SensorRepository sensorRepository;

    /**
     * Configures the mock sensor and returns a hash.
     */
    public String configureMock(Map<String, Object> riverConfig) {
        try {
            // Calculate Hash
            String configJson = objectMapper.writeValueAsString(riverConfig);
            String hash = calculateHash(configJson);

            // Persist to DB
            // We search for a sensor effectively acting as the MOCK one, or create one.
            String mockSensorId = "MOCK-SENSOR-01";
            SensorEntity mockSensor = sensorRepository.findById(mockSensorId)
                    .orElse(SensorEntity.builder()
                            .id(mockSensorId)
                            .name("Mock Sensor Inference")
                            .locationKm(0.0)
                            .strategyType("VIRTUAL")
                            .isActive(true)
                            .build());

            mockSensor.setStrategyType("VIRTUAL");
            mockSensor.setConfiguration(riverConfig);

            sensorRepository.save(mockSensor);
            log.info("Mock Configured with Hash: {} for Sensor {}", hash, mockSensorId);

            return hash;
        } catch (JsonProcessingException e) {
            throw new RuntimeException("Error interacting with Mock Config", e);
        }
    }

    public Map<String, Object> runInference(Map<String, Object> request) {
        // For MVP, if request tells us to use Mock or if system is in Mock profile.
        // Let's implement MOCK response generation
        return generateMockResponse();

        // Real implementation would look like:
        /*
         * return webClient.post()
         * .uri("/predict")
         * .bodyValue(request)
         * .retrieve()
         * .bodyToMono(Map.class)
         * .block();
         */
    }

    private Map<String, Object> generateMockResponse() {
        Map<String, Object> result = new HashMap<>();
        result.put("id", UUID.randomUUID().toString());
        result.put("timestamp", LocalDateTime.now().toString());

        // Summary
        Map<String, Object> summary = new HashMap<>();
        summary.put("message", "Simulated Spill Detected (MOCK)");

        List<Map<String, Object>> hotPoints = new ArrayList<>();
        Map<String, Object> p1 = new HashMap<>();
        p1.put("zone_name", "Mock Zone A");
        p1.put("probability", 0.85);
        p1.put("latitud", 40.030);
        p1.put("longitud", -3.602);
        hotPoints.add(p1);

        summary.put("heat_points", hotPoints);
        result.put("summary", summary);

        // GeoJSON Heatmap
        Map<String, Object> geoJson = new HashMap<>();
        geoJson.put("type", "FeatureCollection");

        List<Map<String, Object>> features = new ArrayList<>();
        // Add random points
        for (int i = 0; i < 10; i++) {
            Map<String, Object> feature = new HashMap<>();
            feature.put("type", "Feature");

            Map<String, Object> geometry = new HashMap<>();
            geometry.put("type", "Point");
            geometry.put("coordinates", Arrays.asList(-3.60 + (Math.random() * 0.01), 40.03 + (Math.random() * 0.01)));
            feature.put("geometry", geometry);

            Map<String, Object> properties = new HashMap<>();
            properties.put("probability", Math.random());
            feature.put("properties", properties);

            features.add(feature);
        }
        geoJson.put("features", features);
        result.put("geojson_heat_map", geoJson);

        return result;
    }

    private String calculateHash(String input) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] encodedhash = digest.digest(input.getBytes(StandardCharsets.UTF_8));
            return bytesToHex(encodedhash);
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException(e);
        }
    }

    private static String bytesToHex(byte[] hash) {
        StringBuilder hexString = new StringBuilder(2 * hash.length);
        for (int i = 0; i < hash.length; i++) {
            String hex = Integer.toHexString(0xff & hash[i]);
            if (hex.length() == 1) {
                hexString.append('0');
            }
            hexString.append(hex);
        }
        return hexString.toString();
    }
}
