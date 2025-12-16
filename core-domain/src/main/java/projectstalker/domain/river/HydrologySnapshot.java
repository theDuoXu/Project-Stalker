package projectstalker.domain.river;

public record HydrologySnapshot(
        float[] temperature,
        float[] ph,
        float[] decay,
        double timeSeconds
) {}