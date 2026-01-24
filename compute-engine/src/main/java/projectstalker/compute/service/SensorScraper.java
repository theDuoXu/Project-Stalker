package projectstalker.compute.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import projectstalker.compute.entity.SensorEntity;
import projectstalker.compute.entity.SensorReadingEntity;
import projectstalker.compute.repository.SensorRepository;
import projectstalker.compute.repository.SensorReadingRepository;
import projectstalker.compute.repository.sql.JpaSensorRepository; // Determine if we use Jpa directly or SensorRepository interface

import java.time.LocalDateTime;
import java.util.List;
import java.util.Random;

@Service
@Slf4j
@RequiredArgsConstructor
public class SensorScraper {

    // We use the JPA repository directly to find all sensors,
    // or use SensorRepository if it supports findAll().
    // SensorRepository interface doesn't have findAll().
    // So we inject JpaSensorRepository or we add findAll to SensorRepository
    // interface.
    // For simplicity, I'll use JpaSensorRepository here or SensorRepositoryImpl if
    // exposed.
    private final JpaSensorRepository jpaSensorRepository;
    private final SensorReadingRepository readingRepository;
    private final RuleEngine ruleEngine; // To trigger alerts

    private final Random random = new Random();

    @Scheduled(cron = "0 0 * * * *") // Every hour
    // @Scheduled(fixedRate = 60000) // For debug: every minute
    public void scrapeSensors() {
        log.info("Starting scheduled sensor scraping...");
        List<SensorEntity> sensors = jpaSensorRepository.findAll();

        for (SensorEntity sensor : sensors) {
            if (Boolean.TRUE.equals(sensor.getIsActive())) {
                try {
                    scrapeAndSave(sensor);
                } catch (Exception e) {
                    log.error("Failed to scrape sensor {}", sensor.getName(), e);
                }
            }
        }
        log.info("Sensor scraping completed.");
    }

    private void scrapeAndSave(SensorEntity sensor) {
        // Mock logic: generate random values closer to realistic ranges based on sensor
        // type ??
        // In real world, we would check sensor.getConfiguration() and call the URL.

        // Simulating READING
        double ph = 7.0 + (random.nextDouble() - 0.5) * 1.0; // 6.5 - 7.5
        double temp = 15.0 + (random.nextDouble() - 0.5) * 5.0; // 12.5 - 17.5

        saveReading(sensor, "ph", ph);
        saveReading(sensor, "temperature", temp);
    }

    private void saveReading(SensorEntity sensor, String parameter, double value) {
        SensorReadingEntity reading = SensorReadingEntity.builder()
                .sensorId(sensor.getId())
                .parameter(parameter)
                .value(value)
                .timestamp(LocalDateTime.now())
                .build();

        readingRepository.save(reading);

        // Trigger Rules
        ruleEngine.evaluate(sensor, parameter, value);
    }
}
