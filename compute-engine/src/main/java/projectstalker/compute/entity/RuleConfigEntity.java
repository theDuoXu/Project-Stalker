package projectstalker.compute.entity;

import jakarta.persistence.*;
import lombok.*;

@Entity
@Table(name = "rule_config")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class RuleConfigEntity {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false, unique = true)
    private String metric; // e.g., "PH", "TEMPERATURE"

    @Column(nullable = false)
    private boolean useLog; // True for Log-Normal distribution

    @Column(nullable = false)
    private double thresholdSigma; // Z-Score threshold (e.g., 3.0)

    @Column(nullable = false)
    private int windowSize; // Rolling window size (e.g., 50)

    @Column(nullable = true)
    private Double minLimit; // Hard minimal limit

    @Column(nullable = true)
    private Double maxLimit; // Hard maximal limit
}
