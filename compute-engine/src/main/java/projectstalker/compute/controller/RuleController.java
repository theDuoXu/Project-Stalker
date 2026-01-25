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

    @GetMapping
    public List<RuleConfigDTO> getAllRules() {
        return repository.findAll().stream()
                .map(e -> new RuleConfigDTO(
                        e.getId(),
                        e.getMetric(),
                        e.isUseLog(),
                        e.getThresholdSigma(),
                        e.getWindowSize()))
                .collect(Collectors.toList());
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

        RuleConfigEntity saved = repository.save(entity);

        return new RuleConfigDTO(
                saved.getId(),
                saved.getMetric(),
                saved.isUseLog(),
                saved.getThresholdSigma(),
                saved.getWindowSize());
    }
}
