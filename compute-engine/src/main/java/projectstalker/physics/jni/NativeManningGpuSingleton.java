// src/main/java/projectstalker/physics/jni/NativeManningGpuSingleton.java
package projectstalker.physics.jni;

import lombok.extern.slf4j.Slf4j;
import projectstalker.utils.NativeLibraryLoader;

import java.nio.FloatBuffer;

@Slf4j
public class NativeManningGpuSingleton implements INativeManningSolver {
    private static volatile NativeManningGpuSingleton INSTANCE = null;

    // --- Carga de la librería nativa (JNI) ---
    static {
        NativeLibraryLoader.loadLibrary("manning_solver");
    }

    private NativeManningGpuSingleton() {
    }

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

    // --- MÉTODOS NATIVOS (Stateful Lifecycle DMA) ---

    /**
     * Inicializa la sesión GPU.
     * Los buffers pasados deben ser DirectBuffers (AllocateDirect).
     */
    @Override
    public native long initSession(
            FloatBuffer bottomWidths,
            FloatBuffer sideSlopes,
            FloatBuffer manningCoefficients,
            FloatBuffer bedSlopes,
            FloatBuffer initialDepths, // Estado Intrinsic
            FloatBuffer initialQ,      // Estado Intrinsic
            int cellCount
    );

    /**
     * Ejecuta el batch utilizando DMA (Zero-Copy).
     * <p>
     * Modificado para DMA y Selección de Estrategia:
     * 1. No devuelve datos, escribe directamente en {@code outputBuffer}.
     * 2. No realiza copias de Java a C++, lee directamente de {@code inputBuffer}.
     * 3. Ambos buffers deben ser DirectBuffers.
     *
     * @param sessionHandle Handle de la sesión C++.
     * @param inputBuffer   Buffer con los nuevos inflows (Input).
     * @param outputBuffer  Buffer donde la GPU escribirá los resultados H y V (Output).
     * @param batchSize     Tamaño del batch a procesar.
     * @param mode          Estrategia de simulación (0=Smart, 1=Full). Ver constantes en {@link INativeManningSolver}.
     * @param stride        Factor de submuestreo.
     * @return 0 si éxito, código de error negativo si falla.
     */
    @Override
    public native int runBatch(
            long sessionHandle,
            FloatBuffer inputBuffer,
            FloatBuffer outputBuffer,
            int batchSize,
            int mode,
            int stride
    );

    @Override
    public native void destroySession(long sessionHandle);

    @Override
    public native int getDeviceCount();
}