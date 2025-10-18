package projectstalker.physics.jni;

// La interfaz que define el contrato nativo
public interface INativeManningSolver {
    float[] solveManningGpu(float[] targetDischarges, float[] initialDepthGuesses, float[] bottomWidths,
                            float[] sideSlopes, float[] manningCoefficients, float[] bedSlopes);

    float[] solveManningGpuBatch(float[] gpuInitialGuess, float[] flatDischargeProfiles, int batchSize, int cellCount,
                                 float[] floats, float[] sideSlopesFP32, float[] manningCoefficientsFP32, float[] bedSlopesFP32);
}