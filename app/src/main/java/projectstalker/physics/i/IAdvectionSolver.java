package projectstalker.physics.i;

import projectstalker.domain.river.RiverGeometry;

public interface IAdvectionSolver extends ISolverComponent {
    /**
     * Calcula la nueva concentración considerando SOLO el movimiento del agua.
     * @param concentration Array actual de concentraciones [mg/L]
     * @param velocity      Array de velocidades [m/s] (Viene de Manning)
     * @param area          Array de áreas mojadas [m2] (Viene de Manning)
     * @param geometry      Geometría (dx)
     * @param dt            Paso de tiempo (segundos)
     * @return Nuevo array de concentraciones.
     */
    float[] solveAdvection(float[] concentration, float[] velocity, float[] area, RiverGeometry geometry, float dt);
}