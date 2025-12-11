package projectstalker.physics.solver;

import projectstalker.config.RiverConfig;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;

@FunctionalInterface
public interface HydrologySolver {
    RiverState calculateNextState(
            RiverState currentState,
            RiverGeometry geometry,
            RiverConfig config,
            double currentTimeInSeconds,
            double inputDischarge
    );
}