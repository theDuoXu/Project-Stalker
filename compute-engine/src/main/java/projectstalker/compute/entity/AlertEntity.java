package projectstalker.compute.entity;

import jakarta.persistence.*;
import lombok.*;

import java.time.LocalDateTime;
import java.util.UUID;

import com.fasterxml.jackson.annotation.JsonAutoDetect;
import com.fasterxml.jackson.annotation.JsonView;
import projectstalker.domain.sensors.SensorViews;

@Entity
@Table(name = "alerts")
@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
@ToString
@JsonAutoDetect(fieldVisibility = JsonAutoDetect.Visibility.ANY)
@JsonView(SensorViews.Public.class)
public class AlertEntity {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    @Column(length = 36)
    private String id;

    @Column(nullable = false)
    private String sensorId;

    @Column(nullable = false)
    private LocalDateTime timestamp;

    @Column(nullable = false)
    @Enumerated(EnumType.STRING)
    private AlertSeverity severity;

    @Column(nullable = false)
    private String message;

    @Column(nullable = false)
    @Enumerated(EnumType.STRING)
    private AlertStatus status; // NEW, ACKNOWLEDGED, CLEARED

    @PrePersist
    protected void onCreate() {
        if (this.timestamp == null) {
            this.timestamp = LocalDateTime.now();
        }
        if (this.status == null) {
            this.status = AlertStatus.NEW;
        }
    }

    public enum AlertSeverity {
        INFO, WARNING, CRITICAL
    }

    public enum AlertStatus {
        NEW, ACKNOWLEDGED, CLEARED
    }
}
