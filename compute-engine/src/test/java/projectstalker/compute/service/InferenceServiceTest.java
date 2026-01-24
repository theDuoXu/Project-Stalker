package projectstalker.compute.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;
import projectstalker.compute.entity.SensorEntity;
import projectstalker.compute.repository.SensorRepository;
import org.springframework.boot.test.autoconfigure.jdbc.AutoConfigureTestDatabase;
import org.springframework.context.annotation.Import;
import projectstalker.compute.TestSecurityConfig;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest
@ActiveProfiles("mock")
@AutoConfigureTestDatabase(replace = AutoConfigureTestDatabase.Replace.ANY)
@Import(TestSecurityConfig.class)
public class InferenceServiceTest {

    @Autowired
    private InferenceService inferenceService;

    @Autowired
    private SensorRepository sensorRepository;

    @Test
    void testConfigureMock_Persistence() {
        System.out.println("SensorRepository Class: " + sensorRepository.getClass().getName());
        Map<String, Object> config = new HashMap<>();
        config.put("riverLength", 1000);
        config.put("name", "Test River");

        String hash = inferenceService.configureMock(config);

        assertNotNull(hash);

        // Verify persistence
        SensorEntity sensor = sensorRepository.findById("MOCK-SENSOR-01").orElse(null);
        assertNotNull(sensor);
        assertEquals("VIRTUAL", sensor.getStrategyType());
        assertNotNull(sensor.getConfiguration());
    }

    @Test
    void testRunInference_MockResponse() {
        Map<String, Object> result = inferenceService.runInference(Map.of());

        assertNotNull(result);
        assertTrue(result.containsKey("summary"));
        assertTrue(result.containsKey("geojson_heat_map"));
    }
}
