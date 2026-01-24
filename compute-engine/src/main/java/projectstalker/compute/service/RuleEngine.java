package projectstalker.compute.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import projectstalker.compute.entity.AlertEntity;
import projectstalker.compute.entity.SensorEntity;
import projectstalker.compute.repository.AlertRepository;

@Service
@Slf4j
@RequiredArgsConstructor
public class RuleEngine {

    private final AlertRepository alertRepository;

    /**
     * Evalúa una lectura entrante y genera alertas si es necesario.
     * Esta versión es básica y verifica límites fijos.
     */
    public void evaluate(SensorEntity sensor, String parameter, double value) {
        // Ejemplo de regla hardcoded: pH fuera de rango [6.5, 9.0]
        if ("ph".equalsIgnoreCase(parameter)) {
            if (value < 6.5 || value > 9.0) {
                createAlert(sensor, AlertEntity.AlertSeverity.WARNING,
                        String.format("pH fuera de rango aceptable: %.2f", value));
            }
        }

        // Ejemplo: Temperatura > 30 grados
        if ("temperature".equalsIgnoreCase(parameter)) {
            if (value > 30.0) {
                createAlert(sensor, AlertEntity.AlertSeverity.CRITICAL,
                        String.format("Temperatura crítica detectada: %.2f", value));
            }
        }
    }

    private void createAlert(SensorEntity sensor, AlertEntity.AlertSeverity severity, String message) {
        AlertEntity alert = AlertEntity.builder()
                .sensorId(sensor.getId())
                .severity(severity)
                .message(message)
                .status(AlertEntity.AlertStatus.NEW)
                .build();
        alertRepository.save(alert);
        log.info("Alert generated for sensor {}: {}", sensor.getName(), message);
    }
}
