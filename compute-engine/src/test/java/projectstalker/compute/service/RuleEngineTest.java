package projectstalker.compute.service;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import projectstalker.compute.entity.AlertEntity;
import projectstalker.compute.entity.RuleConfigEntity;
import projectstalker.compute.entity.SensorEntity;
import projectstalker.compute.repository.AlertRepository;
import projectstalker.compute.repository.RuleConfigRepository;
import projectstalker.compute.repository.SensorReadingRepository;

import java.util.Optional;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.argThat;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class RuleEngineTest {

    @Mock
    private AlertRepository alertRepository;
    @Mock
    private SensorReadingRepository readingRepository;
    @Mock
    private RuleConfigRepository ruleConfigRepository;

    @InjectMocks
    private RuleEngine ruleEngine;

    private SensorEntity sensor;

    @BeforeEach
    void setUp() {
        sensor = SensorEntity.builder().id("sensor-1").name("Sensor 1").build();
    }

    @Test
    void testHardMinLimitExceeded() {
        String metric = "PH";
        RuleConfigEntity config = RuleConfigEntity.builder()
                .metric(metric)
                .minLimit(6.0)
                .maxLimit(8.0)
                .build();

        when(ruleConfigRepository.findByMetric(metric)).thenReturn(Optional.of(config));

        ruleEngine.evaluate(sensor, metric, 5.0);

        verify(alertRepository).save(argThat(alert -> alert.getSeverity() == AlertEntity.AlertSeverity.CRITICAL &&
                alert.getMessage().contains("Valor por debajo del límite físico")));
    }

    @Test
    void testHardMaxLimitExceeded() {
        String metric = "PH";
        RuleConfigEntity config = RuleConfigEntity.builder()
                .metric(metric)
                .minLimit(6.0)
                .maxLimit(8.0)
                .build();

        when(ruleConfigRepository.findByMetric(metric)).thenReturn(Optional.of(config));

        ruleEngine.evaluate(sensor, metric, 9.0);

        verify(alertRepository).save(argThat(alert -> alert.getSeverity() == AlertEntity.AlertSeverity.CRITICAL &&
                alert.getMessage().contains("Valor por encima del límite físico")));
    }

    @Test
    void testWithinLimits() {
        String metric = "PH";
        RuleConfigEntity config = RuleConfigEntity.builder()
                .metric(metric)
                .minLimit(6.0)
                .maxLimit(8.0)
                .thresholdSigma(3.0)
                .windowSize(10)
                .build();

        when(ruleConfigRepository.findByMetric(metric)).thenReturn(Optional.of(config));
        // Empty history so no z-score check
        when(readingRepository.findBySensorIdAndParameterOrderByTimestampDesc(any(), any(), any()))
                .thenReturn(java.util.Collections.emptyList());

        ruleEngine.evaluate(sensor, metric, 7.0);

        verify(alertRepository, never()).save(any());
    }
}
