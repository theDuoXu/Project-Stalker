package projectstalker.physics.jni;

import lombok.extern.slf4j.Slf4j;
import projectstalker.utils.NativeLibraryLoader;

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
        NativeLibraryLoader.loadLibrary("manning_solver");
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