
package projectstalker.domain.dto.twin;

import com.fasterxml.jackson.annotation.JsonView;
import lombok.Builder;
import lombok.With;
import projectstalker.domain.sensors.SensorViews;

/**
 * DTO para crear/editar un Twin
 */
@Builder
@With
@JsonView(SensorViews.Internal.class)
public record FlowPreviewRequest(
        double baseDischarge,      // Caudal base
        double noiseAmplitude,     // Cuánto varía
        float noiseFrequency,      // Cuán rápido varía
        double durationSeconds,    // Cuánto tiempo queremos previsualizar
        int seed                   // Semilla para reproducibilidad
) {}