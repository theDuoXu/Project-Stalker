package projectstalker.compute.entity;

import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;
import projectstalker.config.RiverConfig;
import projectstalker.domain.event.GeologicalEvent;
import projectstalker.domain.river.RiverGeometry;

import java.time.LocalDateTime;
import java.util.List;

/**
 * Representa la persistencia de un Gemelo Digital en la base de datos.
 * Utiliza columnas JSONB de PostgreSQL para almacenar la configuración
 * y la geometría compleja de forma eficiente.
 */
@Entity
@Table(name = "digital_twins")
@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class DigitalTwinEntity {

    @Id
    @Column(length = 36)
    private String id;

    @Column(nullable = false)
    private String name;

    @Column(length = 1000)
    private String description;

    @Column(nullable = false, updatable = false)
    private LocalDateTime createdAt;

    // --- BLOQUES JSON (PostgreSQL JSONB) ---

    /**
     * Almacena el objeto RiverConfig (Record) como un JSON.
     * Esto nos permite guardar la "semilla" y parámetros sin crear 50 columnas.
     */
    @Column(columnDefinition = "jsonb")
    @JdbcTypeCode(SqlTypes.JSON)
    private RiverConfig config;

    /**
     * Almacena la lista de eventos (presas, etc.) como un JSON Array.
     */
    @Column(columnDefinition = "jsonb")
    @JdbcTypeCode(SqlTypes.JSON)
    private List<GeologicalEvent> events;

    /**
     * Almacena la Geometría COMPLETA (arrays de floats) como JSON binario.
     * PostgreSQL comprimirá esto automáticamente (TOAST).
     * Es mucho más rápido leer un solo BLOB JSON que hacer un JOIN de 10.000 filas de puntos.
     */
    @Column(columnDefinition = "jsonb")
    @JdbcTypeCode(SqlTypes.JSON)
    private RiverGeometry geometry;

}