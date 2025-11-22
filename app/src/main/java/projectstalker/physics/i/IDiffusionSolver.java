package projectstalker.physics.i;

import projectstalker.domain.river.RiverGeometry;

public interface IDiffusionSolver {
    /**
     * Calcula la nueva concentración considerando SOLO la dispersión.
     * @param concentration Array actual de concentraciones [mg/L]
     * @param velocity      Array de velocidades [m/s] (Viene de Manning)
     * @param geometry      Geometría (dx)
     * @param dt            Paso de tiempo (segundos)
     * @return Nuevo array de concentraciones.
     */
    float[] solveDiffusion(float[] concentration, float[] velocity, float[] depth, RiverGeometry geometry, float dt);
}