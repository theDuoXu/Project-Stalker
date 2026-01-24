package projectstalker.compute.api;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import projectstalker.compute.entity.SimulationEntity;
import projectstalker.compute.repository.SimulationRepository;
import projectstalker.compute.service.SimulationResultService;
import projectstalker.config.ApiRoutes;
import projectstalker.config.SimulationConfig;
import projectstalker.domain.simulation.IManningResult;
import projectstalker.domain.simulation.SimulationResponseDTO;
import projectstalker.physics.simulator.ManningSimulator;
import projectstalker.compute.service.SimulationService;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.time.LocalDateTime;
import java.util.UUID;

@RestController
@RequestMapping(ApiRoutes.SIMULATIONS)
@Slf4j
@RequiredArgsConstructor
public class SimulationController {

    private final SimulationService simulationService;
    private final SimulationResultService resultService;

    /**
     * Endpoint PRINCIPAL: Lanza la simulación en la GPU.
     * POST https://api.protonenergyindustries/projectstalker/v1/simulation/run
     */
    @PostMapping("/run")
    public ResponseEntity<SimulationResponseDTO> runSimulation(@RequestBody SimulationConfig config) {
        log.info(">>> API: Recibida petición de simulación");
        try {
            SimulationResponseDTO response = simulationService.runSimulation(config);
            return ResponseEntity.ok(response);
        } catch (RuntimeException e) {
            return ResponseEntity.internalServerError().build();
        }
    }

    /**
     * Endpoint SECUNDARIO: Descarga binaria de alta velocidad.
     * GET http://localhost:8080/api/v1/simulation/{id}/binary
     */
    @GetMapping("/{id}/binary")
    public ResponseEntity<byte[]> downloadBinaryResult(@PathVariable String id) {
        IManningResult result = resultService.getResult(id);
        if (result == null) {
            return ResponseEntity.notFound().build();
        }

        try (ByteArrayOutputStream baos = new ByteArrayOutputStream();
                DataOutputStream dos = new DataOutputStream(baos)) {

            // Simple serialization format:
            // 1. Simulation Time (long)
            dos.writeLong(result.getSimulationTime());
            // 2. Timesteps (int)
            dos.writeInt(result.getTimestepCount());
            // 3. Cell Count (int)
            dos.writeInt(result.getGeometry().getCellCount());

            // TODO: Write actual float arrays from result.getStates()
            // This depends on the IManningResult structure. For now, writing header.

            return ResponseEntity.ok()
                    .header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=\"sim_" + id + ".bin\"")
                    .contentType(MediaType.APPLICATION_OCTET_STREAM)
                    .body(baos.toByteArray());
        } catch (IOException e) {
            return ResponseEntity.internalServerError().build();
        }
    }
}