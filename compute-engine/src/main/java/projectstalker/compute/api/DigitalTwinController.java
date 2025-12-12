package projectstalker.compute.api;

import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import projectstalker.compute.service.DigitalTwinService;
import projectstalker.config.RiverConfig;
import projectstalker.domain.dto.twin.FlowPreviewRequest;
import projectstalker.domain.dto.twin.TwinCreateRequest;
import projectstalker.domain.dto.twin.TwinDetailDTO;
import projectstalker.domain.dto.twin.TwinSummaryDTO;

import java.util.List;

@RestController
@RequestMapping("/api/twins")
@RequiredArgsConstructor
public class DigitalTwinController {

    private final DigitalTwinService twinService;

    // =========================================================================
    // 1. GESTIÓN CRUD (Create, Read, Update, Delete)
    // =========================================================================

    // GET /api/twins?limit=5
    // Seguridad: Cubierto por SecurityConfig (GET /api/twins/** -> TECH, ANALYST, ADMIN)
    @GetMapping
    public ResponseEntity<List<TwinSummaryDTO>> getAllTwins(
            @RequestParam(defaultValue = "10") int limit
    ) {
        return ResponseEntity.ok(twinService.getAllTwins(limit));
    }

    // GET /api/twins/{id}
    // Seguridad: Cubierto por SecurityConfig (GET /api/twins/** -> TECH, ANALYST, ADMIN)
    @GetMapping("/{id}")
    public ResponseEntity<TwinDetailDTO> getTwinDetails(@PathVariable String id) {
        return ResponseEntity.ok(twinService.getTwinDetails(id));
    }

    // POST /api/twins
    // Seguridad: Cubierto por SecurityConfig (POST /api/twins/** -> ANALYST, ADMIN)
    @PostMapping
    public ResponseEntity<TwinSummaryDTO> createTwin(@RequestBody TwinCreateRequest request) {
        TwinSummaryDTO newTwin = twinService.createTwin(request);
        return ResponseEntity.status(HttpStatus.CREATED).body(newTwin);
    }

    // PUT /api/twins/{id}
    // Seguridad: Cubierto por SecurityConfig (PUT /api/twins/** -> ANALYST, ADMIN)
    @PutMapping("/{id}")
    public ResponseEntity<TwinDetailDTO> updateTwin(
            @PathVariable String id,
            @RequestBody TwinCreateRequest request
    ) {
        return ResponseEntity.ok(twinService.updateTwin(id, request));
    }

    // DELETE /api/twins/{id}
    // Seguridad: Cubierto por SecurityConfig (DELETE /api/twins/** -> ADMIN)
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteTwin(@PathVariable String id) {
        twinService.deleteTwin(id);
        return ResponseEntity.noContent().build();
    }

    // =========================================================================
    // 2. PREVISUALIZACIÓN DE FÍSICA (Live Preview)
    // =========================================================================
    // Seguridad: Al ser POST, caen en la regla de 'POST /api/twins/**'
    // Por tanto, solo ANALYST y ADMIN pueden ejecutar estas simulaciones efímeras.

    /**
     * Previsualiza el perfil de temperatura espacial del río.
     */
    @PostMapping("/preview/temperature")
    public ResponseEntity<float[]> previewTemperatureModel(
            @RequestParam(defaultValue = "0.0") double timeOfDaySeconds,
            @RequestBody RiverConfig config
    ) {
        float[] profile = twinService.previewTemperature(config, timeOfDaySeconds);
        return ResponseEntity.ok(profile);
    }

    /**
     * Previsualiza el generador de caudal (Hidrograma) en el tiempo.
     */
    @PostMapping("/preview/flow")
    public ResponseEntity<float[]> previewFlowGenerator(
            @RequestBody FlowPreviewRequest request
    ) {
        float[] hydrograph = twinService.previewFlow(request);
        return ResponseEntity.ok(hydrograph);
    }
}