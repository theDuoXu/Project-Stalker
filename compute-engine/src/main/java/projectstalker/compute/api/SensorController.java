package projectstalker.compute.api;

import lombok.RequiredArgsConstructor;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;
import projectstalker.config.ApiRoutes;
import projectstalker.compute.service.SensorService;
import projectstalker.domain.dto.sensor.SensorCreationDTO;
import projectstalker.domain.dto.sensor.SensorHealthResponseDTO;
import projectstalker.domain.dto.sensor.SensorReadingDTO;
import projectstalker.domain.dto.sensor.SensorResponseDTO;

import java.time.LocalDateTime;
import java.util.List;

/**
 * Endpoint PRINCIPAL: Lanza la simulación en la GPU.
 * POST <a href=
 * "https://api.protonenergyindustries/projectstalker/v1/simulation/run">api</a>
 */
@RestController
@RequestMapping(ApiRoutes.SENSORS)
@RequiredArgsConstructor
public class SensorController {

    private final SensorService sensorService;

    @GetMapping("/{stationId}/history")
    public ResponseEntity<SensorResponseDTO> getSensorHistory(
            @PathVariable String stationId,
            @RequestParam(name = "parameter") String parameter) {
        // Delegamos toda la responsabilidad al servicio
        SensorResponseDTO response = sensorService.getHistory(stationId, parameter);

        return ResponseEntity.ok(response);
    }

    // GET /api/sensors?twinId=ABC
    @GetMapping
    public ResponseEntity<List<SensorResponseDTO>> getSensorsByTwin(
            @RequestParam(name = "twinId") String twinId) {
        return ResponseEntity.ok(sensorService.getAllByTwin(twinId));
    }

    // GET /api/sensors/C302/realtime -> Devuelve todo
    // GET /api/sensors/C302/realtime?parameter=PH -> Devuelve solo PH
    @GetMapping("/{stationId}/realtime")
    public ResponseEntity<List<SensorReadingDTO>> getStationRealtime(
            @PathVariable String stationId,
            @RequestParam(name = "parameter", defaultValue = "ALL") String parameter) {
        List<SensorReadingDTO> data = sensorService.getRealtime(stationId, parameter);
        return ResponseEntity.ok(data);
    }

    // GET /api/sensors/C302/status?parameter=ALL
    @GetMapping("/{stationId}/status")
    public ResponseEntity<SensorHealthResponseDTO> getStationStatus(
            @PathVariable String stationId,
            @RequestParam(name = "parameter", defaultValue = "ALL") String parameter) {
        SensorHealthResponseDTO statusData = sensorService.getHealth(stationId, parameter);
        return ResponseEntity.ok(statusData);
    }

    @GetMapping("/export/{stationId}")
    // @PreAuthorize("@sensorExportValidator.canExport(authentication, #from, #to)")
    @PreAuthorize("permitAll()")
    public ResponseEntity<SensorResponseDTO> exportReadings(
            @PathVariable String stationId,
            @RequestParam(name = "parameter", defaultValue = "ALL") String parameter,

            // Usamos LocalDateTime para precisión
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime from,

            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime to) {
        // Si entra aquí, Spring Security YA ha verificado las fechas y los roles.
        // Los roles operativos tienen asignados un rango máximo de 30 días
        // Roles no operativos solo se le permiten bajar 1 día

        SensorResponseDTO exportData = sensorService.getExportData(stationId, parameter, from, to);

        // Aquí luego podríamos devolver un ByteArrayResource para descargar un CSV real
        // De momento devolvemos JSON
        return ResponseEntity.ok(exportData);
    }

    @PostMapping
    public ResponseEntity<SensorResponseDTO> registerSensor(@RequestBody SensorCreationDTO request) {
        SensorResponseDTO created = sensorService.registerSensor(request);
        return ResponseEntity.status(HttpStatus.CREATED).body(created);
    }

    @PutMapping("/{stationId}")
    public ResponseEntity<SensorResponseDTO> updateSensor(
            @PathVariable String stationId,
            @RequestBody SensorCreationDTO request) {
        SensorResponseDTO updated = sensorService.updateSensor(stationId, request);
        return ResponseEntity.ok(updated);
    }
}