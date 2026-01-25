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
    private static final int DEFAULT_WINDOW = 50;

    /**
     * Evalúa una lectura entrante y genera alertas si es necesario.
     */
    public void evaluate(SensorEntity sensor, String parameter, double value) {
        String paramUpper = parameter.toUpperCase();

        // 1. Static Thresholds (Legacy - Keep or move to DB later)
        checkStaticThresholds(sensor, paramUpper, value);

        // 2. Rolling Z-Score (Dynamic Config)
        checkRollingZScore(sensor, paramUpper, value);
    }

    private void checkStaticThresholds(SensorEntity sensor, String parameter, double value) {
        // Keep hardcoded for critical safety limits for now
        if ("PH".equals(parameter)) {
            if (value < 6.5 || value > 9.0) {
                createAlert(sensor, AlertEntity.AlertSeverity.WARNING,
                        String.format("pH fuera de rango aceptable: %.2f", value));
            }
        }
        if ("TEMPERATURE".equals(parameter)) {
            if (value > 35.0) {
                createAlert(sensor, AlertEntity.AlertSeverity.CRITICAL,
                        String.format("Temperatura crítica detectada: %.2f", value));
            }
        }
    }

    private void checkRollingZScore(SensorEntity sensor, String parameter, double currentValue) {
        // Fetch Configuration
        var configOpt = ruleConfigRepository.findByMetric(parameter);

        boolean useLog = configOpt.map(c -> c.isUseLog())
                .orElse("E.COLI".equals(parameter) || "AMMONIUM".equals(parameter)); // Default logic
        double threshold = configOpt.map(c -> c.getThresholdSigma()).orElse(DEFAULT_SIGMA);
        int windowSize = configOpt.map(c -> c.getWindowSize()).orElse(DEFAULT_WINDOW);

        // Fetch History based on Window Size
        java.util.List<projectstalker.compute.entity.SensorReadingEntity> history = readingRepository
                .findBySensorIdAndParameterOrderByTimestampDesc(sensor.getId(), parameter,
                        org.springframework.data.domain.PageRequest.of(0, windowSize));

        if (history.size() < 10)
            return; // Need minimum data points

        double[] values = history.stream()
                .filter(r -> r.getValue() > 0 || !useLog)
                .mapToDouble(r -> useLog ? Math.log(r.getValue()) : r.getValue())
                .toArray();

        if (values.length < 10)
            return;

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
            return;

        // Normalize current
        double currentTransformed = useLog ? (currentValue > 0 ? Math.log(currentValue) : mean) : currentValue;
        double zScore = (currentTransformed - mean) / stdDev;

        if (Math.abs(zScore) > threshold) {
            String msg = String.format("Anomalía Estadística (%s): Valor %.2f (Z-Score: %.1f%s > %.1f)",
                    parameter, currentValue, zScore, useLog ? ", Log-Norm" : "", threshold);
            createAlert(sensor, AlertEntity.AlertSeverity.CRITICAL, msg);
        }
    }

    private void createAlert(SensorEntity sensor, AlertEntity.AlertSeverity severity, String message) {
        AlertEntity alert = AlertEntity.builder()
                .sensorId(sensor.getId())
                .severity(severity)
                .message(message)
                .status(AlertEntity.AlertStatus.NEW)
                .timestamp(java.time.LocalDateTime.now())
                .build();
        alertRepository.save(alert);
        log.info("Alert generated for sensor {}: {}", sensor.getName(), message);
    }
}
