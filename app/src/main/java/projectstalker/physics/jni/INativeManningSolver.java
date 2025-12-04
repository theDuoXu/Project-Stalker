package projectstalker.physics.jni;

import java.nio.FloatBuffer;

// La interfaz que define el contrato nativo
public interface INativeManningSolver {

    /**
     * Lifecycle: Init
     * Carga la geometría estática y el ESTADO INICIAL (Flyweight Intrinsic) en la GPU.
     * Se llama una sola vez (Lazy Init).
     */
    long initSession(
            FloatBuffer bottomWidths,
            FloatBuffer sideSlopes,
            FloatBuffer manningCoefficients,
            FloatBuffer bedSlopes,
            FloatBuffer initialDepths, // Nuevo: Estado base t=0
            FloatBuffer initialQ,      // Nuevo: Estado base t=0
            int cellCount
    );

    /**
     * Lifecycle: Run
     * Ejecuta el batch pasando SOLO los datos dinámicos (Extrinsic State).
     *
     * @param sessionHandle Puntero a la sesión C++.
     * @param newInflows    Array primitivo comprimido [BatchSize] para Pinning rápido.
     * @return Array plano con la matriz cuadrada de resultados [BatchSize * BatchSize * 2].
     */
    float[] runBatch(long sessionHandle, float[] newInflows);

    // Lifecycle: Destroy
    void destroySession(long sessionHandle);
}