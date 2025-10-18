package projectstalker.physics.jni;

import lombok.extern.slf4j.Slf4j;

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
    public native float[] solveManningGpu(float[] targetDischarges, float[] initialDepthGuesses, float[] bottomWidths, float[] sideSlopes, float[] manningCoefficients, float[] bedSlopes);

    public native float[] solveManningGpuBatch(float[] gpuInitialGuess, float[] flatDischargeProfiles, int batchSize, int cellCount, float[] floats, float[] sideSlopesFP32, float[] manningCoefficientsFP32, float[] bedSlopesFP32);

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
