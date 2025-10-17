// Fichero: projectstalker/physics/jni/ManningGpuSolver.java
package projectstalker.physics.jni;

import lombok.extern.slf4j.Slf4j;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;
import projectstalker.physics.model.FlowProfileModel;

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

     *
     * @param initialGuess profundidades iniciales (calculadas secuencialmente hasta llenar el río) para facilitar newton raphson
     * @param allDischargeProfiles allDischargeProfiles Target para cada paso de simulación
     * @param geometry geometría básica del río
     * @return  [][0] son las nuevas profundidades y [][1] son las nuevas velocidades.
     */
    public static double[][][] solveBatch(double[] initialGuess,double[][] allDischargeProfiles, RiverGeometry geometry) {
        RiverGeometry.ManningGpuRiverGeometryFP32 gpuGeometry = createGpuGeometry(geometry);
        int batchSize = allDischargeProfiles.length;
        int cellCount = allDischargeProfiles[0].length;
        
        float[] gpuInitialGuess = sanitizeInitialDepths(initialGuess);
        float[][] gpuDischargeProfiles = sanitizeDischargeProfiles(allDischargeProfiles);

        float [] gpuResults = NativeManningGpuSingleton.getInstance().solveManningGpuBatch();
        
        return unpackGpuResults(gpuResults, batchSize, cellCount);
    }

    /**
     * Desempaqueta el array plano de resultados de la GPU a la estructura tridimensional de Java.
     *
     * @param gpuResults Array plano (1D) que contiene todos los resultados. Se asume que la estructura
     * está intercalada: [prof_t0_c0, vel_t0_c0, prof_t0_c1, vel_t0_c1, ..., prof_t1_c0, vel_t1_c0, ...]
     * @param batchSize  El número de pasos de simulación.
     * @param cellCount  El número de celdas por paso.
     * @return Un array tridimensional [batchSize][2][cellCount] donde [][0] son profundidades y [][1] son velocidades.
     */
    private static double[][][] unpackGpuResults(float[] gpuResults, int batchSize, int cellCount) {
        // 1. Validación CRÍTICA: Asegurarse de que el tamaño del array de entrada es el esperado.
        int expectedSize = batchSize * cellCount * 2; // 2 porque tenemos profundidad y velocidad por cada celda.
        if (gpuResults == null || gpuResults.length != expectedSize) {
            throw new IllegalArgumentException(String.format(
                    "Error al desempaquetar resultados de GPU. Tamaño de array inesperado. Esperado: %d, Recibido: %d",
                    expectedSize, gpuResults != null ? gpuResults.length : 0
            ));
        }

        // 2. Crear la estructura de destino.
        // Dimensiones: [paso de tiempo][tipo de dato (0=profundidad, 1=velocidad)][celda]
        double[][][] results = new double[batchSize][2][cellCount];

        // 3. Iterar y rellenar la estructura de destino.
        for (int i = 0; i < batchSize; i++) { // Bucle sobre cada paso de simulación (tiempo).

            // Calculamos el punto de inicio en el array plano para este paso de simulación.
            int timeStepOffset = i * cellCount * 2;

            for (int j = 0; j < cellCount; j++) { // Bucle sobre cada celda del río.

                // Calculamos el índice para la profundidad y velocidad de la celda 'j' en el paso 'i'.
                int sourceIndexForDepth = timeStepOffset + j * 2;
                int sourceIndexForVelocity = sourceIndexForDepth + 1;

                // Asignar los valores, convirtiendo de float a double.
                results[i][0][j] = gpuResults[sourceIndexForDepth]; // [i]=tiempo, [0]=profundidades, [j]=celda
                results[i][1][j] = gpuResults[sourceIndexForVelocity]; // [i]=tiempo, [1]=velocidades, [j]=celda
            }
        }

        return results;
    }


    // --- Métodos de Ayuda para la Preparación de Datos ---

    private static float[] precomputeTargetDischarges(int cellCount, RiverState currentState, RiverGeometry geometry, FlowProfileModel flowGenerator, double targetTimeInSeconds) {
        float[] discharges = new float[cellCount];
        discharges[0] = (float) flowGenerator.getDischargeAt(targetTimeInSeconds);
        for (int i = 1; i < cellCount; i++) {
            double prevArea = geometry.getCrossSectionalArea(i - 1, currentState.getWaterDepthAt(i - 1));
            discharges[i] = (float) (prevArea * currentState.getVelocityAt(i - 1));
        }
        return discharges;
    }
    private static float[][] sanitizeDischargeProfiles(double[][] allDischargeProfiles) {
        int batchSize = allDischargeProfiles.length;
        int cellCount = allDischargeProfiles[0].length;

        // Crear el nuevo array de floats con las mismas dimensiones.
        float[][] sanitizedDischarges = new float[batchSize][cellCount];

        // Iterar sobre cada paso de simulación (dimensión exterior).
        for (int i = 0; i < batchSize; i++) {
            // Iterar sobre cada celda del río en ese paso (dimensión interior).
            for (int j = 0; j < cellCount; j++) {
                double originalDischarge = allDischargeProfiles[i][j];

                // Aplicar la lógica de saneamiento:
                // Si el caudal es 0 o negativo, establecer un valor mínimo.
                if (originalDischarge <= 0) {
                    sanitizedDischarges[i][j] = 0.001f;
                } else {
                    // Si es positivo, simplemente convertirlo a float.
                    sanitizedDischarges[i][j] = (float) originalDischarge;
                }
            }
        }

        return sanitizedDischarges;
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