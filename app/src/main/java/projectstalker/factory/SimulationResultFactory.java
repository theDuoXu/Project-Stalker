package projectstalker.factory;

import projectstalker.config.SimulationConfig;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;
import projectstalker.domain.simulation.*;

import java.util.List;

/**
 * Fábrica centralizada para construir la implementación correcta de {@link IManningResult}.
 * <p>
 * Decide qué estructura de datos utilizar basándose en la estrategia de simulación (CPU/GPU),
 * la configuración de memoria (Smart/Full Evolution) y el formato de los datos recibidos.
 */
public class SimulationResultFactory {

    // --- 1. RESULTADOS CPU (Legacy / Validación) ---

    /**
     * Crea un resultado DENSO para simulaciones de CPU.
     * Útil cuando el solver acumula objetos RiverState paso a paso en una lista Java.
     */
    public static DenseManningResult createCpuResult(
            RiverGeometry geometry,
            List<RiverState> states,
            long executionTimeMs) {

        return DenseManningResult.builder()
                .geometry(geometry)
                .states(states)
                .simulationTime(executionTimeMs)
                .build();
    }

    // --- 2. RESULTADOS GPU SMART (Flyweight) ---

    /**
     * Crea un resultado FLYWEIGHT optimizado para la estrategia GPU SMART.
     * <p>
     * Se usa cuando el solver devuelve "Deltas" (Olas) sobre un río estacionario.
     * Requiere el estado inicial para poder reconstruir la simulación.
     */
    /**
     * Crea un resultado FLYWEIGHT optimizado.
     * MODIFICADO: Ahora acepta arrays planos (packed) directamente.
     */
    public static FlyweightManningResult createSmartGpuResult(
            SimulationConfig config,
            RiverGeometry geometry,
            RiverState initialState,
            float[] packedDepths,
            float[] packedVelocities,
            float[][][] auxData,
            long executionTimeMs) {

        return FlyweightManningResult.builder()
                .geometry(geometry)
                .simulationTime(executionTimeMs)
                .initialState(initialState)
                .packedDepths(packedDepths)
                .packedVelocities(packedVelocities)
                .auxData(auxData)
                .build();
    }

    // --- 3. RESULTADOS GPU FULL EVOLUTION (Strided / Packed) ---

    /**
     * Crea un resultado STRIDED (o denso empacado) para la estrategia GPU FULL EVOLUTION.
     * <p>
     * Se usa cuando el solver devuelve arrays planos gigantes que contienen toda la historia.
     * Es más eficiente que {@link DenseManningResult} porque usa arrays primitivos.
     */
    public static StridedManningResult createStridedGpuResult(
            SimulationConfig config,
            RiverGeometry geometry,
            float[] packedDepths,
            float[] packedVelocities,
            int logicTimestepCount,
            long executionTimeMs) {

        return StridedManningResult.builder()
                .geometry(geometry)
                .simulationTime(executionTimeMs)
                .packedDepths(packedDepths)
                .packedVelocities(packedVelocities)
                .strideFactor(config.getGpuFullEvolutionStride()) // Dato crítico de la config
                .logicTimestepCount(logicTimestepCount)
                .build();
    }

    // --- 4. RESULTADOS GPU BIG DATA (Chunked) ---

    /**
     * Crea un resultado CHUNKED (Paginado) para simulaciones masivas.
     * <p>
     * Se usa cuando los datos vienen troceados en listas de arrays para evitar
     * desbordar el tamaño máximo de array de Java o gestionar streaming.
     */
    public static ChunkedManningResult createChunkedGpuResult(
            SimulationConfig config,
            RiverGeometry geometry,
            List<float[]> depthChunks,
            List<float[]> velocityChunks,
            int stepsPerChunk,
            int logicTimestepCount,
            long executionTimeMs) {

        return ChunkedManningResult.builder()
                .geometry(geometry)
                .simulationTime(executionTimeMs)
                .depthChunks(depthChunks)
                .velocityChunks(velocityChunks)
                .stepsPerChunk(stepsPerChunk)
                .strideFactor(config.getGpuFullEvolutionStride())
                .logicTimestepCount(logicTimestepCount)
                .build();
    }
}