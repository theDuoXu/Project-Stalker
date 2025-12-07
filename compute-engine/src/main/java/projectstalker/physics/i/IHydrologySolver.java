package projectstalker.physics.i;

import projectstalker.config.RiverConfig;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;

@FunctionalInterface
public interface IHydrologySolver {
    RiverState calculateNextState(
            RiverState currentState,
            RiverGeometry geometry,
            RiverConfig config,
            double currentTimeInSeconds,
            double inputDischarge
    );
}