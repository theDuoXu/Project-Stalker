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
public class ManningGpuSolver {

    // --- Carga de la librería nativa (JNI) ---
    static {
        if ("true".equalsIgnoreCase(System.getProperty("projectstalker.native.enabled"))) {
            try {
                log.info("Native library loading ENABLED. Attempting to load 'manning_solver'...");
                System.loadLibrary("manning_solver");
                log.info("Native library 'manning_solver' loaded successfully.");
            } catch (UnsatisfiedLinkError e) {
                log.error("FATAL: Native library 'manning_solver' failed to load. Ensure the library is in the java.library.path.", e);
                System.exit(1);
            }
        } else {
            log.warn("Native library loading is DISABLED. Running in pure Java/mock mode.");
        }
    }

    /**
     * Resuelve un paso de tiempo completo en la GPU.
     *
     * @param currentState     El estado actual del río.
     * @param geometry         La geometría del río.
     * @param flowGenerator    El generador de caudales para obtener el caudal de entrada.
     * @param currentTimeInSeconds El tiempo actual de la simulación.
     * @return Un array de arrays de doubles donde [0] son las nuevas profundidades y [1] son las nuevas velocidades.
     */
    public double[][] solve(RiverState currentState, RiverGeometry geometry, FlowProfileModel flowGenerator, double currentTimeInSeconds) {
        final int cellCount = geometry.getCellCount();

        // 1. Preparar y sanitizar todos los datos para la GPU.
        float[] targetDischarges = precomputeTargetDischarges(cellCount, currentState, geometry, flowGenerator, currentTimeInSeconds);
        float[] initialDepthGuesses = sanitizeInitialDepths(cellCount, currentState);
        RiverGeometry.ManningGpuRiverGeometryFP32 gpuGeometry = createGpuGeometry(geometry);

        // 2. Llamada al método nativo.
        float[] gpuResults = solveManningGpu(
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
     * Declaración del JNI nativo que se comunica con la librería C++/CUDA.
     */
    private native float[] solveManningGpu(float[] targetDischarges, float[] initialDepthGuesses, float[] bottomWidths, float[] sideSlopes, float[] manningCoefficients, float[] bedSlopes);

    // --- Métodos de Ayuda para la Preparación de Datos ---

    private float[] precomputeTargetDischarges(int cellCount, RiverState currentState, RiverGeometry geometry, FlowProfileModel flowGenerator, double currentTime) {
        float[] discharges = new float[cellCount];
        discharges[0] = (float) flowGenerator.getDischargeAt(currentTime);
        for (int i = 1; i < cellCount; i++) {
            double prevArea = geometry.getCrossSectionalArea(i - 1, currentState.getWaterDepthAt(i - 1));
            discharges[i] = (float) (prevArea * currentState.getVelocityAt(i - 1));
        }
        return discharges;
    }

    private float[] sanitizeInitialDepths(int cellCount, RiverState currentState) {
        float[] depths = new float[cellCount];
        for (int i = 0; i < cellCount; i++) {
            double d = currentState.getWaterDepthAt(i);
            depths[i] = (d <= 1e-3) ? 0.001f : (float) d;
        }
        return depths;
    }

    private RiverGeometry.ManningGpuRiverGeometryFP32 createGpuGeometry(RiverGeometry geometry) {
        return new RiverGeometry.ManningGpuRiverGeometryFP32(
                toFloatArray(geometry.cloneBottomWidth()),
                toFloatArray(geometry.cloneSideSlope()),
                toFloatArray(geometry.cloneManningCoefficient()),
                calculateAndSanitizeBedSlopes(geometry)
        );
    }

    private float[] calculateAndSanitizeBedSlopes(RiverGeometry geometry) {
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

    private double[][] unpackGpuResults(float[] gpuResults, int cellCount) {
        double[] newWaterDepth = new double[cellCount];
        double[] newVelocity = new double[cellCount];

        for (int i = 0; i < cellCount; i++) {
            newWaterDepth[i] = gpuResults[i * 2];
            newVelocity[i] = gpuResults[i * 2 + 1];
        }
        return new double[][]{newWaterDepth, newVelocity};
    }

    private float[] toFloatArray(double[] doubleArray) {
        float[] floatArray = new float[doubleArray.length];
        for (int i = 0; i < doubleArray.length; i++) {
            floatArray[i] = (float) doubleArray[i];
        }
        return floatArray;
    }
}