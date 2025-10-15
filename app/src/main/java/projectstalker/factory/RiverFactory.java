package projectstalker.factory;

import projectstalker.config.RiverConfig;
import projectstalker.domain.river.InitialRiver;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;
import projectstalker.physics.solver.IHydrologySolver; // Importar el solver

public class RiverFactory {

    private final RiverGeometryFactory geometryFactory;
    private final IHydrologySolver hydrologySolver; // Nueva dependencia

    // Inyectamos ambas dependencias
    public RiverFactory(RiverGeometryFactory geometryFactory, IHydrologySolver hydrologySolver) {
        this.geometryFactory = geometryFactory;
        this.hydrologySolver = hydrologySolver;
    }

    /**
     * Crea una geometría de río y la simula hasta alcanzar un estado estacionario
     * con un caudal de entrada constante.
     *
     * @param config La configuración para generar la geometría del río.
     * @param initialDischarge El caudal constante (en m³/s) para estabilizar el río.
     * @return Un objeto InitialRiver que contiene tanto la geometría como el estado estable.
     */
    public InitialRiver createStableRiver(RiverConfig config, double initialDischarge) {
        // 1. Crear la geometría del río
        RiverGeometry geometry = geometryFactory.createRealisticRiver(config);
        int cellCount = geometry.getCellCount();

        // 2. Simular el llenado desde un cauce seco
        RiverState currentState = new RiverState(new double[cellCount], new double[cellCount], new double[cellCount], new double[cellCount]);

        long timeInSeconds = 0;
        int iterations = 0;
        int maxIterations = cellCount * 2; // Límite de seguridad

        // Bucle hasta que el agua llegue al final
        while (currentState.getWaterDepthAt(cellCount - 1) < 0.01) {
            currentState = hydrologySolver.calculateNextState(currentState, geometry, config, timeInSeconds, initialDischarge);
            timeInSeconds += 1; // Avanzamos 1 segundo por paso (puede eliminarse, es para tener una mayor variedad)
            iterations++;
            if (iterations > maxIterations) {
                // En una aplicación real, aquí lanzarías una excepción o registrarías un error.
                System.err.println("Advertencia: La estabilización del río superó el máximo de iteraciones.");
                break;
            }
        }

        System.out.printf("Río estabilizado con %.1f m³/s después de %d iteraciones.\n", initialDischarge, iterations);

        // 3. Devolver el par de geometría y estado estable
        return new InitialRiver(geometry, currentState);
    }
}