package projectstalker.physics.i;

import projectstalker.domain.river.RiverGeometry;

public interface IReactionSolver {
    /**
     * Calcula la nueva concentración considerando SOLO la química.
     */
    float[] solveReaction(float[] concentration, RiverGeometry geometry, float dt);
}