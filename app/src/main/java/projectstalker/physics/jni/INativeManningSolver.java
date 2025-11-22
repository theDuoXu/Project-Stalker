package projectstalker.physics.jni;

import java.nio.FloatBuffer;

// La interfaz que define el contrato nativo
public interface INativeManningSolver {
    float[] solveManningGpu(float[] targetDischarges, float[] initialDepthGuesses, FloatBuffer bottomWidths,
                            FloatBuffer sideSlopes, FloatBuffer manningCoefficients, FloatBuffer bedSlopes);

    float[] solveManningGpuBatch(float[] gpuInitialGuess, float[] flatDischargeProfiles, int batchSize, int cellCount,
                                 FloatBuffer bottomWidths, FloatBuffer sideSlopesFP32, FloatBuffer manningCoefficientsFP32, FloatBuffer bedSlopesFP32);
}