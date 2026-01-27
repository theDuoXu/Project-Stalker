package projectstalker.compute.service;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.orm.jpa.DataJpaTest;
import projectstalker.compute.entity.SensorReadingEntity;
import projectstalker.compute.repository.SensorReadingRepository;

import java.time.LocalDateTime;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

@DataJpaTest
public class PersistenceIntegrationTest {

    @Autowired
    private SensorReadingRepository readingRepository;

    @Test
    void testSaveAndFind() {
        String sensorId = "73ebe32c-abd6-42c4-9636-64608fe828cd";
        String stationId = sensorId;

        // 1. Save
        SensorReadingEntity entity = SensorReadingEntity.builder()
                .sensorId(stationId)
                .parameter("PH")
                .timestamp(LocalDateTime.now())
                .value(7.5)
                .build();

        readingRepository.save(entity);

        // 2. Find
        List<SensorReadingEntity> found = readingRepository.findTop10BySensorIdOrderByTimestampDesc(sensorId);

        assertThat(found).isNotEmpty();
        assertThat(found.get(0).getSensorId()).isEqualTo(sensorId);

        System.out.println("Integration Test: Found " + found.size() + " readings.");
    }
}
