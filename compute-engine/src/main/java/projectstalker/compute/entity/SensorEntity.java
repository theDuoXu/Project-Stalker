package projectstalker.compute.entity;

import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;
import projectstalker.domain.sensors.SensorType;

import java.time.LocalDateTime;
import java.util.Map;

/**
 * Representa un sensor (físico o virtual) ubicado en un tramo del río.
 * Utiliza UUID para identidad técnica y JSONB para la configuración polimórfica (Strategy Pattern).
 */
@Entity
@Table(name = "sensors")
@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class SensorEntity {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    @Column(length = 36)
    private String id;

    /**
     * Identificador amigable para el usuario (ej: "S-CABECERA-01").
     * No es la PK, pero debe ser buscable.
     */
    @Column(nullable = false)
    private String name;

    /**
     * Tipo de variable que mide (PH, TEMPERATURE, FLOW, etc.).
     * Guardamos el String (código) para evitar problemas si reordenamos el Enum.
     */
    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private SensorType type;

    /**
     * Ubicación relativa en el tramo del río (Kilómetro exacto).
     * Vital para pintarlo en el mapa o calcular la dispersión aguas abajo.
     */
    @Column(nullable = false)
    private Double locationKm;

    /**
     * Discriminador de la Estrategia: "VIRTUAL" o "REAL".
     * Ayuda al backend a saber qué fábrica usar para instanciar el comportamiento.
     */
    @Column(nullable = false)
    private String strategyType;

    /**
     * CONFIGURACIÓN POLIMÓRFICA (JSONB).
     * Aquí reside la magia del Patrón Estrategia en base de datos.
     * - Si strategyType="VIRTUAL": Guarda coeficientes, semillas, pgvector simulado.
     * - Si strategyType="REAL": Guarda URLs, Webhooks, Topics MQTT, API Keys.
     */
    @Column(columnDefinition = "jsonb")
    @JdbcTypeCode(SqlTypes.JSON)
    private Map<String, Object> configuration;

    @Column(nullable = false, updatable = false)
    private LocalDateTime createdAt;

    @Column(nullable = false)
    private Boolean isActive;

    // --- RELACIONES ---

    /**
     * Relación con el Gemelo Digital (El tramo del río).
     * Un sensor pertenece a UN tramo.
     * Fetch Lazy para no traerse todo el río (geometría pesada) si solo consultamos el sensor.
     */
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "twin_id", nullable = false)
    private DigitalTwinEntity twin;

    // --- LIFECYCLE CALLBACKS ---

    @PrePersist
    protected void onCreate() {
        if (this.createdAt == null) {
            this.createdAt = LocalDateTime.now();
        }
        if (this.isActive == null) {
            this.isActive = true;
        }
    }
}