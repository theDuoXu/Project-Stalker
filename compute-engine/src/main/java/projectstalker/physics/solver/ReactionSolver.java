package projectstalker.physics.solver;

import projectstalker.domain.river.RiverGeometry;

public interface ReactionSolver extends SolverComponent {
    /**
     * Calcula la nueva concentración considerando SOLO la química.
     */
    float[] solveReaction(float[] concentration, float[] temperature, RiverGeometry geometry, float dt);
}