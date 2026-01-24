package projectstalker.compute.entity;

import jakarta.persistence.*;
import lombok.*;

import java.time.LocalDateTime;

@Entity
@Table(name = "simulations")
@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class SimulationEntity {

    @Id
    @Column(length = 36)
    private String id; // UUID assigned by Controller

    @Column(nullable = false)
    private String digitalTwinId; // ID of the river config/twin used

    @Column(nullable = false)
    @Enumerated(EnumType.STRING)
    private SimulationStatus status;

    @Column(nullable = false, updatable = false)
    private LocalDateTime createdAt;

    private LocalDateTime finishedAt;

    @PrePersist
    protected void onCreate() {
        if (this.createdAt == null) {
            this.createdAt = LocalDateTime.now();
        }
    }

    public enum SimulationStatus {
        PENDING, RUNNING, COMPLETED, FAILED
    }
}
