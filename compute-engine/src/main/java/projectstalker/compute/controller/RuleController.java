package projectstalker.compute.controller;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;
import projectstalker.compute.entity.RuleConfigEntity;
import projectstalker.compute.entity.SensorEntity;
import projectstalker.compute.entity.SensorReadingEntity;
import projectstalker.compute.repository.RuleConfigRepository;
import projectstalker.domain.dto.rule.RuleConfigDTO;

import java.util.List;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/rules")
@Slf4j
@RequiredArgsConstructor
@CrossOrigin(origins = "*") // Allow JavaFX client
public class RuleController {

        private final RuleConfigRepository repository;

        private static final java.util.List<String> DEFAULT_METRICS = java.util.List.of(
                        "PH", "TEMPERATURE", "E.COLI", "AMMONIUM", "FLOW", "CONDUCTIVITY", "DISSOLVED_OXYGEN",
                        "TURBIDITY");

        @GetMapping
        public List<RuleConfigDTO> getAllRules() {
                if (repository.count() == 0) {
                        log.info("No rules found. Seeding defaults...");
                        seedDefaults();
                }

                return repository.findAll().stream()
                                .map(e -> new RuleConfigDTO(
                                                e.getId(),
                                                e.getMetric(),
                                                e.isUseLog(),
                                                e.getThresholdSigma(),
                                                e.getWindowSize(),
                                                e.getMinLimit(),
                                                e.getMaxLimit()))
                                .collect(Collectors.toList());
        }

        private void seedDefaults() {
                List<RuleConfigEntity> defaults = new java.util.ArrayList<>();
                for (String metric : DEFAULT_METRICS) {
                        boolean isLog = "E.COLI".equals(metric) || "AMMONIUM".equals(metric) || "FLOW".equals(metric)
                                        || "CONDUCTIVITY".equals(metric);
                        defaults.add(RuleConfigEntity.builder()
                                        .metric(metric)
                                        .useLog(isLog)
                                        .thresholdSigma(4.0)
                                        .windowSize(3)
                                        .build());
                }
                repository.saveAll(defaults);
        }

        @PostMapping
        public RuleConfigDTO saveRule(@RequestBody RuleConfigDTO dto) {
                log.info("BACKEND RECEIVED DTO: {}", dto);
                log.info("Saving rule for metric: {}", dto.getMetric());

                RuleConfigEntity entity = repository.findByMetric(dto.getMetric())
                                .orElse(new RuleConfigEntity());

                entity.setMetric(dto.getMetric());
                entity.setUseLog(dto.isUseLog());
                entity.setThresholdSigma(dto.getThresholdSigma());
                entity.setWindowSize(dto.getWindowSize());
                entity.setMinLimit(dto.getMinLimit());
                entity.setMaxLimit(dto.getMaxLimit());

                RuleConfigEntity saved = repository.save(entity);

                // Trigger Re-evaluation
                reprocessHistory(saved.getMetric());

                return new RuleConfigDTO(
                                saved.getId(),
                                saved.getMetric(),
                                saved.isUseLog(),
                                saved.getThresholdSigma(),
                                saved.getWindowSize(),
                                saved.getMinLimit(),
                                saved.getMaxLimit());
        }

        private final projectstalker.compute.service.RuleEngine ruleEngine;
        private final projectstalker.compute.repository.sql.JpaSensorRepository sensorRepository;
        private final projectstalker.compute.repository.SensorReadingRepository readingRepository;

        private void reprocessHistory(String metric) {
                log.info("Reprocessing history for metric: {} (Last 24h)", metric);
                java.time.LocalDateTime cutoff = java.time.LocalDateTime.now().minusHours(24);

                // Get distinct sensor IDs that actually have readings in the database
                List<String> sensorIds = readingRepository.findDistinctSensorIds();
                log.info("Found {} distinct sensor IDs with readings", sensorIds.size());

                for (String sensorId : sensorIds) {
                        // Fetch readings (Case Insensitive for "PH" vs "pH")
                        var readings = readingRepository.findBySensorIdAndParameterIgnoreCaseAndTimestampAfter(
                                        sensorId, metric, cutoff);

                        if (readings.isEmpty()) {
                                continue; // Skip sensors with no readings for this metric
                        }

                        log.info("Sensor {}: Found {} readings for {}", sensorId, readings.size(), metric);

                        // Try to find matching SensorEntity for context (optional)
                        var sensorOpt = sensorRepository.findById(sensorId);
                        SensorEntity sensor = sensorOpt.orElse(null);

                        int alertsGenerated = 0;
                        for (var r : readings) {
                                var alert = ruleEngine.evaluate(sensor, metric, r.getValue(), r.getTimestamp());
                                if (alert.isPresent()) {
                                        alertsGenerated++;
                                        log.info("Generated Historical Alert: {} at {}", alert.get().getMessage(),
                                                        r.getTimestamp());
                                }
                        }
                        if (alertsGenerated > 0) {
                                log.info("Sensor {}: Generated {} alerts during reprocessing", sensorId,
                                                alertsGenerated);
                        }
                }
                log.info("Reprocessing complete for {}", metric);
        }
}
