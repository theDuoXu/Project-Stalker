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
    private final projectstalker.compute.repository.SensorReadingRepository readingRepository;

    // Parameters that require log-normalization (distributions like concentration,
    // flow)
    private static final java.util.Set<String> LOG_NORMAL_PARAMS = java.util.Set.of("E.COLI", "AMMONIUM", "FLOW",
            "CONDUCTIVITY");

    /**
     * Evalúa una lectura entrante y genera alertas si es necesario.
     */
    public void evaluate(SensorEntity sensor, String parameter, double value) {
        String paramUpper = parameter.toUpperCase();

        // 1. Static Thresholds (Legacy/Simple)
        checkStaticThresholds(sensor, paramUpper, value);

        // 2. Rolling Z-Score (Statistical Anomaly Detection)
        checkRollingZScore(sensor, paramUpper, value);
    }

    private void checkStaticThresholds(SensorEntity sensor, String parameter, double value) {
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
        // Fetch last 50 readings for statistical context
        java.util.List<projectstalker.compute.entity.SensorReadingEntity> history = readingRepository
                .findTop50BySensorIdAndParameterOrderByTimestampDesc(sensor.getId(), parameter);

        if (history.size() < 10)
            return; // Need minimum data points

        boolean useLog = LOG_NORMAL_PARAMS.contains(parameter);

        // Extract values (skip current one if it was just saved? Usually scraper saves
        // then evaluates.
        // If so, the current value influences the mean slightly, which is fine, or we
        // exclude it.)
        // Assuming evaluate is called AFTER save, history[0] might be current. Let's
        // exclude it to compare against "past".

        double[] values = history.stream()
                .filter(r -> r.getValue() > 0 || !useLog) // Log requires > 0
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

        if (Math.abs(zScore) > 4.0) {
            String msg = String.format("Anomalía Estadística (%s): Valor %.2f (Z-Score: %.1f%s)",
                    parameter, currentValue, zScore, useLog ? ", Log-Norm" : "");
            createAlert(sensor, AlertEntity.AlertSeverity.CRITICAL, msg);
        }
    }

    private void createAlert(SensorEntity sensor, AlertEntity.AlertSeverity severity, String message) {
        AlertEntity alert = AlertEntity.builder()
                .sensorId(sensor.getId())
                .severity(severity)
                .message(message)
                .status(AlertEntity.AlertStatus.NEW)
                .timestamp(java.time.LocalDateTime.now()) // Ensure timestamp is set
                .build();
        alertRepository.save(alert);
        log.info("Alert generated for sensor {}: {}", sensor.getName(), message);
    }
}
