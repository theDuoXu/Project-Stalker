package projectstalker.physics.model;
@FunctionalInterface
public interface TimeEvolutionModel {
    public float[] generateProfile(double currentTimeInSeconds);
}
