package projectstalker.compute.api;

import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import projectstalker.compute.api.config.ApiRoutes;
import projectstalker.config.SimulationConfig;
import projectstalker.domain.simulation.SimulationResponseDTO;
import projectstalker.domain.simulation.IManningResult;
import projectstalker.physics.simulator.ManningSimulator;

import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

@RestController
@RequestMapping(ApiRoutes.SIMULATIONS)
@Slf4j
public class SimulationController {

    // Simulación de "Persistencia en Memoria" temporal para la demo.
    // Guardamos los resultados aquí para que el cliente los descargue en binario después.
    private final ConcurrentHashMap<String, IManningResult> resultCache = new ConcurrentHashMap<>();

    /**
     * Endpoint PRINCIPAL: Lanza la simulación en la GPU.
     * POST https://api.protonenergyindustries/projectstalker/v1/simulation/run
     */
    @PostMapping("/run")
    public ResponseEntity<SimulationResponseDTO> runSimulation(@RequestBody SimulationConfig config) {
        log.info(">>> API: Recibida petición de simulación (Steps: {}, GPU: {})",
                config.getTotalTimeSteps(), config.isUseGpuAccelerationOnManning());

        try {
            // 1. Instanciar el Simulador (Crea el BatchProcessor y conecta con JNI)
            // Usamos try-with-resources para asegurar que se libera la memoria nativa al terminar si falla algo
            // PERO: El resultado lo queremos mantener en caché, así que cuidado con cerrar buffers si son off-heap.
            // En tu caso, IManningResult copia los datos a Java Heap, así que es seguro cerrar el simulador.

            IManningResult result;
            try (ManningSimulator simulator = new ManningSimulator(config.getRiverConfig(), config)) {
                result = simulator.runFullSimulation();
            }

            // 2. Guardar en Caché Temporal (simulando Redis)
            String simId = UUID.randomUUID().toString();
            resultCache.put(simId, result);

            // 3. Devolver Resumen JSON
            SimulationResponseDTO response = new SimulationResponseDTO(
                    result.getSimulationTime(),
                    result.getTimestepCount(),
                    result.getGeometry().getCellCount(),
                    "COMPLETED",
                    "/api/v1/simulation/" + simId + "/binary" // Link para descarga
            );

            return ResponseEntity.ok(response);

        } catch (Exception e) {
            log.error("Error ejecutando simulación", e);
            return ResponseEntity.internalServerError().build();
        }
    }

    /**
     * Endpoint SECUNDARIO: Descarga binaria de alta velocidad.
     * GET http://localhost:8080/api/v1/simulation/{id}/binary
     * (Implementación pendiente: Aquí volcaremos los float[] al OutputStream)
     */
    @GetMapping("/{id}/binary")
    public ResponseEntity<byte[]> downloadBinaryResult(@PathVariable String id) {
        // TODO: Implementar serialización binaria rápida
        return ResponseEntity.notFound().build();
    }
}