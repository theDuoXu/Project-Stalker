package projectstalker.compute.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import projectstalker.compute.entity.SimulationEntity;
import projectstalker.compute.repository.SimulationRepository;
import projectstalker.config.SimulationConfig;
import projectstalker.domain.simulation.IManningResult;
import projectstalker.domain.simulation.SimulationResponseDTO;
import projectstalker.physics.simulator.ManningSimulator;

import java.time.LocalDateTime;
import java.util.UUID;

@Service
@Slf4j
@RequiredArgsConstructor
public class SimulationService {

    private final SimulationResultService resultService;
    private final SimulationRepository simulationRepository;
    private final SimulatorFactory simulatorFactory;

    /**
     * Executes the simulation based on the provided configuration.
     * Currently runs synchronously (blocking).
     *
     * @param config The simulation configuration.
     * @return SimulationResponseDTO with the result summary.
     */
    public SimulationResponseDTO runSimulation(SimulationConfig config) {
        log.info("Starting simulation service (Steps: {}, GPU: {})",
                config.getTotalTimeSteps(), config.isUseGpuAccelerationOnManning());

        // 1. Create Simulation Entity (Audit Log)
        String simId = UUID.randomUUID().toString();
        SimulationEntity entity = SimulationEntity.builder()
                .id(simId)
                // In a real app, extract Digital Twin ID from config or context
                .digitalTwinId("TWIN-DEFAULT")
                .status(SimulationEntity.SimulationStatus.RUNNING)
                .build();
        simulationRepository.save(entity);

        try {
            // 2. Select and Run Simulator Strategy
            // Currently supporting ManningSimulator.
            // TODO: Add logic to switch to TransportSimulator if config requests it.

            IManningResult result;
            // Using try-with-resources to ensure native resources are freed
            try (ManningSimulator simulator = simulatorFactory.createManningSimulator(config)) {
                result = simulator.runFullSimulation();
            }

            // 3. Save Result to Redis
            resultService.saveResult(simId, result);

            // 4. Update Entity Status
            entity.setStatus(SimulationEntity.SimulationStatus.COMPLETED);
            entity.setFinishedAt(LocalDateTime.now());
            simulationRepository.save(entity);

            log.info("Simulation {} completed successfully.", simId);

            // 5. Return Summary
            return new SimulationResponseDTO(
                    result.getSimulationTime(),
                    result.getTimestepCount(),
                    result.getGeometry().getCellCount(),
                    "COMPLETED",
                    "/api/v1/simulation/" + simId + "/binary");

        } catch (Exception e) {
            log.error("Simulation {} failed", simId, e);
            entity.setStatus(SimulationEntity.SimulationStatus.FAILED);
            entity.setFinishedAt(LocalDateTime.now());
            simulationRepository.save(entity);
            throw new RuntimeException("Simulation execution failed", e);
        }
    }
}
