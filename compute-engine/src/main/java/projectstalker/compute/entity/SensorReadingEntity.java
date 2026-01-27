package projectstalker.compute.entity;

import jakarta.persistence.*;
import lombok.*;

import java.time.LocalDateTime;

@Entity
@Table(name = "sensor_readings")
@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class SensorReadingEntity {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    @Column(length = 36)
    private String id;

    @Column(name = "sensor_id", nullable = false)
    private String sensorId;

    @Column(nullable = false)
    private LocalDateTime timestamp;

    @Column(nullable = false)
    private String parameter; // e.g. "ph", "temperature"

    @Column(nullable = false)
    private Double value;

    @PrePersist
    protected void onCreate() {
        if (this.timestamp == null) {
            this.timestamp = LocalDateTime.now();
        }
    }
}
