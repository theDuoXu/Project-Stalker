package projectstalker.compute.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import projectstalker.compute.entity.SensorEntity;
import projectstalker.compute.entity.SensorReadingEntity;
import projectstalker.compute.repository.SensorReadingRepository;
import projectstalker.compute.repository.sql.JpaSensorRepository;
import org.springframework.web.reactive.function.client.WebClient;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Random;

@Service
@Slf4j
@RequiredArgsConstructor
public class SensorScraper {

    private final projectstalker.compute.service.SensorService sensorService;
    private final JpaSensorRepository jpaSensorRepository;
    private final SensorReadingRepository readingRepository;
    private final RuleEngine ruleEngine;
    private final WebClient.Builder webClientBuilder;

    // We lazily build or just build in constructor if builder is available
    private WebClient webClient;

    private final Random random = new Random();

    // Constructor handles initialization

    @jakarta.annotation.PostConstruct
    public void init() {
        this.webClient = webClientBuilder.build();
    }

    // Run every 10 seconds for Demo purposes (Realtime feedback)
    // Run every 10 seconds for Demo purposes (Realtime feedback)
    @Scheduled(fixedDelay = 10000)
    public void scrapeSensors() {
        // log.debug("Starting scheduled sensor scraping...");
        // Commented out to avoid spamming logs too much, but useful if nothing works

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
    }

    private void scrapeAndSave(SensorEntity sensor) {
        if ("REAL_IoT_WEBHOOK".equals(sensor.getStrategyType())) {
            scrapeRealWebhook(sensor);
        } else {
            // For virtual sensors, we don't necessarily need to "scrape" and save endlessly
            // if we are generating on fly, BUT existing logic saves them to DB.
            // Let's keep it for now.
            scrapeMock(sensor);
        }
    }

    private void scrapeRealWebhook(SensorEntity sensor) {
        try {
            // Delegate to SensorService which has robust SAICA parsing and persistence
            log.debug("Delegating scraping for {} to SensorService...", sensor.getName());

            // "ALL" fetches all parameters found in the webhook
            List<projectstalker.domain.dto.sensor.SensorReadingDTO> readings = sensorService.getRealtime(sensor.getId(),
                    "ALL");

            for (var r : readings) {
                // Readings are ALREADY saved to DB by SensorService.fetchAndParseSaica
                // We just need to trigger Rule Engine
                ruleEngine.evaluate(sensor, r.tag(), r.value());
            }

            if (!readings.isEmpty()) {
                log.info("Scraped and processed {} readings for {}", readings.size(), sensor.getName());
            }

        } catch (Exception e) {
            log.error("Failed to process Real Webhook for {}", sensor.getName(), e);
        }
    }

    private void scrapeMock(SensorEntity sensor) {
        if ("VIRTUAL_SINE".equals(sensor.getStrategyType())) {
            scrapeSineWave(sensor);
        } else {
            // Default random noise if no strategy matches
            double ph = 7.0 + (random.nextDouble() - 0.5) * 1.0;
            double temp = 15.0 + (random.nextDouble() - 0.5) * 5.0;
            saveReading(sensor, "ph", ph);
            saveReading(sensor, "temperature", temp);
        }
    }

    private void scrapeSineWave(SensorEntity sensor) {
        try {
            java.util.Map<String, Object> config = sensor.getConfiguration();
            double offset = getDouble(config, "offset", 10.0);
            double amplitude = getDouble(config, "amplitude", 5.0);
            double frequency = getDouble(config, "frequency", 0.1);

            // Calculate sine value based on current time
            double timeSeconds = System.currentTimeMillis() / 1000.0;
            double value = offset + amplitude * Math.sin(2 * Math.PI * frequency * timeSeconds);

            // Add some noise
            value += (random.nextDouble() - 0.5) * (amplitude * 0.1);

            saveReading(sensor, "value", value);
        } catch (Exception e) {
            log.error("Error generating sine wave for {}", sensor.getName(), e);
        }
    }

    private double getDouble(java.util.Map<String, Object> map, String key, double def) {
        if (map == null || !map.containsKey(key))
            return def;
        Object val = map.get(key);
        if (val instanceof Number)
            return ((Number) val).doubleValue();
        try {
            return Double.parseDouble(val.toString());
        } catch (Exception e) {
            return def;
        }
    }

    private void saveReading(SensorEntity sensor, String parameter, double value) {
        SensorReadingEntity reading = SensorReadingEntity.builder()
                .sensorId(sensor.getId())
                .parameter(parameter)
                .value(value)
                .timestamp(LocalDateTime.now())
                .build();

        readingRepository.save(reading);
        ruleEngine.evaluate(sensor, parameter, value);
    }
}
