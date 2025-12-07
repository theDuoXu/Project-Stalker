package projectstalker.physics.jni;

import lombok.extern.slf4j.Slf4j;

import java.nio.FloatBuffer;

/**
 * Singleton que carga la librería nativa y expone los métodos JNI para Transporte.
 * NO usar directamente en lógica de negocio; usar a través de GpuMusclTransportSolver.
 */
@Slf4j
public class NativeTransportGpuSingleton implements INativeTransportSolver {

    private static NativeTransportGpuSingleton instance;

    // Bloque estático para cargar la librería una sola vez
    static {
        try {
            if (Boolean.getBoolean("projectstalker.native.enabled")) {
                System.loadLibrary("manning_solver"); // Usamos la misma librería compartida
                log.info("Native library 'manning_solver' loaded successfully for Transport.");
            } else {
                log.warn("Native library loading DISABLED via system property.");
            }
        } catch (UnsatisfiedLinkError e) {
            log.error("Failed to load native library 'manning_solver'. GPU Transport will fail.", e);
        }
    }

    private NativeTransportGpuSingleton() {}

    public static synchronized NativeTransportGpuSingleton getInstance() {
        if (instance == null) {
            instance = new NativeTransportGpuSingleton();
        }
        return instance;
    }

    @Override
    public native float[] solveTransportEvolution(
            FloatBuffer cInBuf,
            FloatBuffer uBuf,
            FloatBuffer hBuf,
            FloatBuffer areaBuf,
            FloatBuffer tempBuf,
            FloatBuffer alphaBuf,
            FloatBuffer decayBuf,
            float dx,
            float dtSub,
            int numSteps,
            int cellCount
    );
}