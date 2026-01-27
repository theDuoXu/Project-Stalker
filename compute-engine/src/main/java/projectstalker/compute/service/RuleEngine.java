package projectstalker.compute.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import projectstalker.compute.entity.AlertEntity;
import projectstalker.compute.entity.SensorEntity;
import projectstalker.compute.repository.AlertRepository;
import projectstalker.compute.repository.SensorReadingRepository;

@Service
@Slf4j
@RequiredArgsConstructor
public class RuleEngine {

    private final AlertRepository alertRepository;
    private final SensorReadingRepository readingRepository;
    private final projectstalker.compute.repository.RuleConfigRepository ruleConfigRepository;

    // Default Fallbacks
    private static final double DEFAULT_SIGMA = 4.0;
    private static final int DEFAULT_WINDOW = 5;

    /**
     * Evalúa una lectura entrante y genera alertas si es necesario.
     */
    /**
     * Evalúa una lectura entrante y genera alertas si es necesario.
     */
    public java.util.Optional<AlertEntity> evaluate(SensorEntity sensor, String parameter, double value) {
        return evaluate(sensor, parameter, value, java.time.LocalDateTime.now());
    }

    public java.util.Optional<AlertEntity> evaluate(SensorEntity sensor, String parameter, double value,
            java.time.LocalDateTime timestamp) {
        String paramUpper = parameter.toUpperCase();

        // 1. Hard Limits (Physics)
        var alert = checkHardLimits(sensor, paramUpper, value, timestamp);
        if (alert.isPresent())
            return alert;

        // 2. Rolling Z-Score (Dynamic Config)
        return checkRollingZScore(sensor, paramUpper, value, timestamp);
    }

    private java.util.Optional<AlertEntity> createAlert(SensorEntity sensor, AlertEntity.AlertSeverity severity,
            String message, String metric,
            java.time.LocalDateTime eventTime) {
        // Idempotency Check: 1 alert per metric per sensor per day
        java.time.LocalDateTime startOfDay = java.time.LocalDate.now().atStartOfDay();
        boolean exists = alertRepository.existsBySensorIdAndMetricAndTimestampAfter(sensor.getId(), metric, startOfDay);

        if (exists && severity != AlertEntity.AlertSeverity.INFO) {
            log.debug("Duplicate alert suppressed for {} {}: {}", sensor.getName(), metric, message);
            return java.util.Optional.empty();
        }

        AlertEntity alert = AlertEntity.builder()
                .sensorId(sensor.getId())
                .metric(metric)
                .severity(severity)
                .message(message)
                .status(AlertEntity.AlertStatus.NEW)
                .timestamp(eventTime != null ? eventTime : java.time.LocalDateTime.now())
                .build();
        return java.util.Optional.of(alertRepository.save(alert));
    }

    private java.util.Optional<AlertEntity> checkHardLimits(SensorEntity sensor, String parameter, double value,
            java.time.LocalDateTime timestamp) {
        var configOpt = ruleConfigRepository.findByMetric(parameter);

        // Check Min Limit
        if (configOpt.isPresent() && configOpt.get().getMinLimit() != null) {
            if (value < configOpt.get().getMinLimit()) {
                return createAlert(sensor, AlertEntity.AlertSeverity.CRITICAL,
                        String.format("Valor por debajo del límite físico (%s): %.2f < %.2f",
                                parameter, value, configOpt.get().getMinLimit()),
                        parameter, timestamp);
            }
        }

        // Check Max Limit
        if (configOpt.isPresent() && configOpt.get().getMaxLimit() != null) {
            if (value > configOpt.get().getMaxLimit()) {
                return createAlert(sensor, AlertEntity.AlertSeverity.CRITICAL,
                        String.format("Valor por encima del límite físico (%s): %.2f > %.2f",
                                parameter, value, configOpt.get().getMaxLimit()),
                        parameter, timestamp);
            }
        }

        return java.util.Optional.empty();
    }

    private java.util.Optional<AlertEntity> checkRollingZScore(SensorEntity sensor, String parameter,
            double currentValue,
            java.time.LocalDateTime timestamp) {
        // Fetch Configuration
        var configOpt = ruleConfigRepository.findByMetric(parameter);

        boolean useLog = configOpt.map(c -> c.isUseLog())
                .orElse("E.COLI".equals(parameter) || "AMMONIUM".equals(parameter)); // Default logic
        double threshold = configOpt.map(c -> c.getThresholdSigma()).orElse(DEFAULT_SIGMA);
        int windowSize = configOpt.map(c -> c.getWindowSize() > 0 ? c.getWindowSize() : DEFAULT_WINDOW)
                .orElse(DEFAULT_WINDOW);

        // Fetch History based on Window Size
        java.util.List<projectstalker.compute.entity.SensorReadingEntity> history = readingRepository
                .findBySensorIdAndParameterIgnoreCaseOrderByTimestampDesc(sensor.getId(), parameter,
                        org.springframework.data.domain.PageRequest.of(0, windowSize));

        if (history.size() < 10)
            return java.util.Optional.empty(); // Need minimum data points

        double[] values = history.stream()
                .filter(r -> r.getValue() > 0 || !useLog)
                .mapToDouble(r -> useLog ? Math.log(r.getValue()) : r.getValue())
                .toArray();

        if (values.length < 10)
            return java.util.Optional.empty();

        // Calculate Stats
        double mean = 0.0;
        for (double v : values)
            mean += v;
        mean /= values.length;

        double sumSq = 0.0;
        for (double v : values)
            sumSq += (v - mean) * (v - mean);
        double stdDev = Math.sqrt(sumSq / values.length);

        if (stdDev == 0)
            return java.util.Optional.empty();

        // Normalize current
        double currentTransformed = useLog ? (currentValue > 0 ? Math.log(currentValue) : mean) : currentValue;
        double zScore = (currentTransformed - mean) / stdDev;

        if (Math.abs(zScore) > threshold) {
            String msg = String.format("Anomalía Estadística (%s): Valor %.2f (Z-Score: %.1f%s > %.1f)",
                    parameter, currentValue, zScore, useLog ? ", Log-Norm" : "", threshold);
            return createAlert(sensor, AlertEntity.AlertSeverity.CRITICAL, msg, parameter, timestamp);
        }

        return java.util.Optional.empty();
    }

}
