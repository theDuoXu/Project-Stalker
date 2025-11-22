package projectstalker.physics.i;

import projectstalker.domain.river.RiverGeometry;

public interface IReactionSolver extends ISolverComponent {
    /**
     * Calcula la nueva concentración considerando SOLO la química.
     */
    float[] solveReaction(float[] concentration, float[] temperature, RiverGeometry geometry, float dt);
}