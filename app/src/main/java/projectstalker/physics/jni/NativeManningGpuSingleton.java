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

    // --- MÉTODOS NATIVOS (Stateful Lifecycle Actualizado) ---

    /**
     * Inicializa la sesión.
     * Firma actualizada: Ahora acepta 6 Buffers (4 geometría + 2 estado inicial).
     */
    @Override
    public native long initSession(
            FloatBuffer bottomWidths,
            FloatBuffer sideSlopes,
            FloatBuffer manningCoefficients,
            FloatBuffer bedSlopes,
            FloatBuffer initialDepths, // Nuevo
            FloatBuffer initialQ,      // Nuevo
            int cellCount
    );

    /**
     * Ejecuta el batch.
     * Firma actualizada: Solo recibe el handle y los nuevos inflows (Flyweight).
     */
    @Override
    public native float[] runBatch(long sessionHandle, float[] newInflows);

    @Override
    public native void destroySession(long sessionHandle);
}