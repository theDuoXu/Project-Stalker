package projectstalker.domain.dto.rule;

public record RuleConfigDTO(
        Long id,
        String metric,
        boolean useLog,
        double thresholdSigma,
        int windowSize) {
}
