package projectstalker.physics.jni;

import lombok.extern.slf4j.Slf4j;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;
import projectstalker.physics.model.FlowProfileModel;

import java.util.Arrays;

/**
 * Actúa como un puente (wrapper) hacia la librería nativa JNI para resolver
 * la ecuación de Manning en la GPU. Encapsula toda la lógica de carga,
 * preparación de datos y llamada al código nativo.
 */
@Slf4j
public final class ManningGpuSolver {

    private ManningGpuSolver() {
    }

    /**
     * Resuelve un paso de tiempo completo en la GPU.
     *
     * @param currentState        El estado actual del río.
     * @param geometry            La geometría del río.
     * @param flowGenerator       El generador de caudales para obtener el caudal de entrada.
     * @param targetTimeInSeconds El tiempo actual de la simulación.
     * @return Un array de arrays de doubles donde [0] son las nuevas profundidades y [1] son las nuevas velocidades.
     */
    public static double[][] solve(RiverState currentState, RiverGeometry geometry, FlowProfileModel flowGenerator, double targetTimeInSeconds) {
        final int cellCount = geometry.getCellCount();

        // 1. Preparar y sanitizar todos los datos para la GPU.
        float[] targetDischarges = precomputeTargetDischarges(cellCount, currentState, geometry, flowGenerator, targetTimeInSeconds);
        float[] initialDepthGuesses = sanitizeInitialDepths(cellCount, currentState);
        RiverGeometry.ManningGpuRiverGeometryFP32 gpuGeometry = createGpuGeometry(geometry);

        // 2. Llamada al método nativo.
        float[] gpuResults = NativeManningGpuSingleton.getInstance().solveManningGpu(
                targetDischarges,
                initialDepthGuesses,
                gpuGeometry.bottomWidthsFP32(),
                gpuGeometry.sideSlopesFP32(),
                gpuGeometry.manningCoefficientsFP32(),
                gpuGeometry.bedSlopesFP32()
        );

        // 3. Desempaquetar los resultados a un formato útil para Java.
        return unpackGpuResults(gpuResults, cellCount);
    }

    /**
     * Resuelve un lote de pasos de tiempo completo en la GPU.
     *
     * @param initialGuess          Profundidades iniciales para el primer paso del batch.
     * @param allDischargeProfiles  Caules de entrada para cada paso del batch.
     * @param geometry              Geometría básica del río.
     * @return [batchSize][0][cellCount] son las nuevas profundidades y [batchSize][1][cellCount] son las nuevas velocidades.
     */
    public static double[][][] solveBatch(double[] initialGuess, double[][] allDischargeProfiles, RiverGeometry geometry) {

        // 1. Parámetros de control
        int batchSize = allDischargeProfiles.length;
        int cellCount = allDischargeProfiles[0].length;

        // 2. Preparar y sanitizar los datos
        float[] gpuInitialGuess = sanitizeInitialDepths(initialGuess);

        // Aplanar y sanear el array 2D de caudales a 1D - Mayor rendimiento GPU
        float[] flatDischargeProfiles = flattenDischargeProfiles(allDischargeProfiles);

        // 3. Preparar la geometría
        RiverGeometry.ManningGpuRiverGeometryFP32 gpuGeometry = createGpuGeometry(geometry);

        // 4. Llamada al método nativo con la firma completa.
        float [] gpuResults = NativeManningGpuSingleton.getInstance().solveManningGpuBatch(
                gpuInitialGuess,
                flatDischargeProfiles,
                batchSize,
                cellCount,
                gpuGeometry.bottomWidthsFP32(),
                gpuGeometry.sideSlopesFP32(),
                gpuGeometry.manningCoefficientsFP32(),
                gpuGeometry.bedSlopesFP32()
        );

        // 5. Desempaquetar los resultados
        return unpackGpuResults(gpuResults, batchSize, cellCount);
    }

    /**
     * Implementa el aplanamiento (flattening) de un array 2D de caudales a un array 1D
     * y realiza la sanitización/conversión a float.
     * La estructura de aplanamiento es: [Q_t0_c0, Q_t0_c1, ..., Q_t1_c0, Q_t1_c1, ...]
     * * @param allDischargeProfiles Array 2D [batchSize][cellCount]
     * @return Array 1D aplanado [batchSize * cellCount]
     */
    private static float[] flattenDischargeProfiles(double[][] allDischargeProfiles) {
        int batchSize = allDischargeProfiles.length;
        int cellCount = allDischargeProfiles[0].length;
        int totalSize = batchSize * cellCount;

        float[] flatDischarges = new float[totalSize];

        // Iterar sobre el tiempo (paso de batch)
        for (int i = 0; i < batchSize; i++) {
            // Iterar sobre el espacio (celda)
            for (int j = 0; j < cellCount; j++) {
                double originalDischarge = allDischargeProfiles[i][j];
                int flatIndex = i * cellCount + j; // Cálculo del índice 1D

                // Sanitización
                if (originalDischarge <= 0) {
                    flatDischarges[flatIndex] = 0.001f;
                } else {
                    flatDischarges[flatIndex] = (float) originalDischarge;
                }
            }
        }

        return flatDischarges;
    }


    /**
     * Desempaqueta el array plano de resultados de la GPU a la estructura tridimensional de Java. (Lógica existente)
     */
    private static double[][][] unpackGpuResults(float[] gpuResults, int batchSize, int cellCount) {
        // ... (Lógica de desempaquetado mantenida e intacta) ...
        int expectedSize = batchSize * cellCount * 2;
        if (gpuResults == null || gpuResults.length != expectedSize) {
            throw new IllegalArgumentException(String.format(
                    "Error al desempaquetar resultados de GPU. Tamaño de array inesperado. Esperado: %d, Recibido: %d",
                    expectedSize, gpuResults != null ? gpuResults.length : 0
            ));
        }

        double[][][] results = new double[batchSize][2][cellCount];

        for (int i = 0; i < batchSize; i++) {
            int timeStepOffset = i * cellCount * 2;

            for (int j = 0; j < cellCount; j++) {
                int sourceIndexForDepth = timeStepOffset + j * 2;
                int sourceIndexForVelocity = sourceIndexForDepth + 1;

                results[i][0][j] = gpuResults[sourceIndexForDepth];
                results[i][1][j] = gpuResults[sourceIndexForVelocity];
            }
        }

        return results;
    }


    // --- Métodos de Ayuda para la Preparación de Datos (Resto de métodos existentes) ---

    private static float[] precomputeTargetDischarges(int cellCount, RiverState currentState, RiverGeometry geometry, FlowProfileModel flowGenerator, double targetTimeInSeconds) {
        float[] discharges = new float[cellCount];
        discharges[0] = (float) flowGenerator.getDischargeAt(targetTimeInSeconds);
        for (int i = 1; i < cellCount; i++) {
            double prevArea = geometry.getCrossSectionalArea(i - 1, currentState.getWaterDepthAt(i - 1));
            discharges[i] = (float) (prevArea * currentState.getVelocityAt(i - 1));
        }
        return discharges;
    }

    private static float[] sanitizeInitialDepths(double[] initialGuess) {
        float[] depths = new float[initialGuess.length];
        for (int i = 0; i < initialGuess.length; i++) {
            double d = initialGuess[i];
            depths[i] = (d <= 1e-3) ? 0.001f : (float) d;
        }
        return depths;
    }

    private static float[] sanitizeInitialDepths(int cellCount, RiverState currentState) {
        float[] depths = new float[cellCount];
        for (int i = 0; i < cellCount; i++) {
            double d = currentState.getWaterDepthAt(i);
            depths[i] = (d <= 1e-3) ? 0.001f : (float) d;
        }
        return depths;
    }

    private static RiverGeometry.ManningGpuRiverGeometryFP32 createGpuGeometry(RiverGeometry geometry) {
        return new RiverGeometry.ManningGpuRiverGeometryFP32(
                toFloatArray(geometry.cloneBottomWidth()),
                toFloatArray(geometry.cloneSideSlope()),
                toFloatArray(geometry.cloneManningCoefficient()),
                calculateAndSanitizeBedSlopes(geometry)
        );
    }

    private static float[] calculateAndSanitizeBedSlopes(RiverGeometry geometry) {
        double[] elevations = geometry.cloneElevationProfile();
        double dx = geometry.getDx();
        int cellCount = geometry.getCellCount();
        float[] sanitizedSlopes = new float[cellCount];

        for (int i = 0; i < cellCount - 1; i++) {
            double slope = (elevations[i] - elevations[i + 1]) / dx;
            sanitizedSlopes[i] = (slope <= 1e-7) ? 1e-7f : (float) slope;
        }

        if (cellCount > 1) {
            sanitizedSlopes[cellCount - 1] = sanitizedSlopes[cellCount - 2];
        } else if (cellCount == 1) {
            sanitizedSlopes[0] = 1e-7f;
        }
        return sanitizedSlopes;
    }

    private static double[][] unpackGpuResults(float[] gpuResults, int cellCount) {
        double[] newWaterDepth = new double[cellCount];
        double[] newVelocity = new double[cellCount];

        for (int i = 0; i < cellCount; i++) {
            newWaterDepth[i] = gpuResults[i * 2];
            newVelocity[i] = gpuResults[i * 2 + 1];
        }
        return new double[][]{newWaterDepth, newVelocity};
    }

    private static float[] toFloatArray(double[] doubleArray) {
        float[] floatArray = new float[doubleArray.length];
        for (int i = 0; i < doubleArray.length; i++) {
            floatArray[i] = (float) doubleArray[i];
        }
        return floatArray;
    }
}