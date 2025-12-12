package projectstalker.physics.model;

public interface TimeSeriesGenerator {
    public default float[] generateProfile(double durationInSeconds, double timeStepInSeconds) {
        return generateProfile(0, durationInSeconds, timeStepInSeconds);
    }

    public float[] generateProfile(double startTimeInSeconds ,double durationInSeconds, double timeStepInSeconds);
}
