package projectstalker.domain.simulation;

public record SimulationResponseDTO(
        long executionTimeMs,
        int totalSteps,
        int cellCount,
        String status,
        String downloadUrl
) {}