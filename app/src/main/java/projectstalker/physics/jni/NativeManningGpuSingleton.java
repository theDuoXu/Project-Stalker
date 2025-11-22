package projectstalker.physics.jni;

import lombok.extern.slf4j.Slf4j;

import java.nio.FloatBuffer;

@Slf4j
public class NativeManningGpuSingleton implements INativeManningSolver{
    private static volatile NativeManningGpuSingleton INSTANCE = null;

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

    private NativeManningGpuSingleton() {
    }

    /**
     * Declaración del JNI nativo que se comunica con la librería C++/CUDA.
     */
    public native float[] solveManningGpu(float[] targetDischarges, float[] initialDepthGuesses, FloatBuffer bottomWidths, FloatBuffer sideSlopes, FloatBuffer manningCoefficients, FloatBuffer bedSlopes);

    public native float[] solveManningGpuBatch(float[] gpuInitialGuess, float[] flatDischargeProfiles, int batchSize, int cellCount, FloatBuffer bottomWidths, FloatBuffer sideSlopesFP32, FloatBuffer manningCoefficientsFP32, FloatBuffer bedSlopesFP32);

    public static NativeManningGpuSingleton getInstance() {
        if (INSTANCE == null) {
            synchronized (NativeManningGpuSingleton.class) {
                if (INSTANCE == null) {
                    INSTANCE = new NativeManningGpuSingleton();
                }
            }
        }
        return INSTANCE;
    }
}
