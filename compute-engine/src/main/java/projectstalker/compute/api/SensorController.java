package projectstalker.compute.api;

import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import projectstalker.compute.service.SensorService;
import projectstalker.domain.sensors.SensorResponseDTO;

@RestController
@RequestMapping("/api/sensors")
@RequiredArgsConstructor // Lombok genera el constructor para la inyección de dependencias
public class SensorController {

    private final SensorService sensorService;

    @GetMapping("/{stationId}/history")
    public ResponseEntity<SensorResponseDTO> getSensorHistory(
            @PathVariable("stationId") String stationId,
            @RequestParam(name = "parameter") String parameter
    ) {
        // Delegamos toda la responsabilidad al servicio
        SensorResponseDTO response = sensorService.getHistory(stationId, parameter);

        return ResponseEntity.ok(response);
    }
}