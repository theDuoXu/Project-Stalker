// src/main/java/projectstalker/physics/jni/INativeManningSolver.java
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
     * Lifecycle: Run (DMA / Zero-Copy)
     * Ejecuta el batch pasando SOLO los datos dinámicos (Extrinsic State).
     * <p>
     * Ahora opera "in-place" sobre memoria Pinned.
     *
     * @param sessionHandle Puntero a la sesión C++.
     * @param inputBuffer   Buffer Directo con los nuevos inflows (Input).
     * @param outputBuffer  Buffer Directo donde escribir resultados (Output).
     * @param batchSize     Tamaño del batch a procesar.
     * @return Código de estado (0 = éxito).
     */
    int runBatch(
            long sessionHandle,
            FloatBuffer inputBuffer,
            FloatBuffer outputBuffer,
            int batchSize
    );

    // Lifecycle: Destroy
    void destroySession(long sessionHandle);
}