package projectstalker.compute.controller;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;
import projectstalker.compute.entity.RuleConfigEntity;
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
                log.info("Saving rule for metric: {}", dto.metric());

                RuleConfigEntity entity = repository.findByMetric(dto.metric())
                                .orElse(new RuleConfigEntity());

                entity.setMetric(dto.metric());
                entity.setUseLog(dto.useLog());
                entity.setThresholdSigma(dto.thresholdSigma());
                entity.setWindowSize(dto.windowSize());
                entity.setMinLimit(dto.minLimit());
                entity.setMaxLimit(dto.maxLimit());

                RuleConfigEntity saved = repository.save(entity);

                return new RuleConfigDTO(
                                saved.getId(),
                                saved.getMetric(),
                                saved.isUseLog(),
                                saved.getThresholdSigma(),
                                saved.getWindowSize(),
                                saved.getMinLimit(),
                                saved.getMaxLimit());
        }
}
