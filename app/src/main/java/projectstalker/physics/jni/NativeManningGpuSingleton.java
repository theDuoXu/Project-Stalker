// src/main/java/projectstalker/physics/jni/NativeManningGpuSingleton.java
package projectstalker.physics.jni;

import lombok.extern.slf4j.Slf4j;

import java.nio.FloatBuffer;

@Slf4j
public class NativeManningGpuSingleton implements INativeManningSolver {
    private static volatile NativeManningGpuSingleton INSTANCE = null;

    // --- Carga de la librería nativa (JNI) ---
    static {
        // Obtenemos la propiedad, default a "false" si es nula
        String enabledProp = System.getProperty("projectstalker.native.enabled");
        if ("true".equalsIgnoreCase(enabledProp)) {
            try {
                log.info("Native library loading ENABLED. Attempting to load 'manning_solver'...");
                System.loadLibrary("manning_solver");
                log.info("Native library 'manning_solver' loaded successfully.");
            } catch (UnsatisfiedLinkError e) {
                log.error("FATAL: Native library 'manning_solver' failed to load. Ensure the library is in the java.library.path.", e);
                // No hacemos System.exit(1) para permitir que los tests unitarios continúen si fallan
                throw new RuntimeException("Fallo crítico cargando librería nativa", e);
            }
        } else {
            log.warn("Native library loading is DISABLED. Running in pure Java/mock mode.");
        }
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
     * @return 0 si éxito, código de error negativo si falla.
     */
    @Override
    public native int runBatch(
            long sessionHandle,
            FloatBuffer inputBuffer,
            FloatBuffer outputBuffer,
            int batchSize,
            int mode
    );

    @Override
    public native void destroySession(long sessionHandle);
}