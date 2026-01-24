package projectstalker.compute.service;

import org.springframework.stereotype.Component;
import projectstalker.config.SimulationConfig;
import projectstalker.physics.simulator.ManningSimulator;

@Component
public class SimulatorFactory {

    public ManningSimulator createManningSimulator(SimulationConfig config) {
        return new ManningSimulator(config.getRiverConfig(), config);
    }

    // Future: createTransportSimulator
}
