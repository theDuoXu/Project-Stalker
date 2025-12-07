// src/main/java/projectstalker/physics/jni/INativeManningSolver.java
package projectstalker.physics.jni;

import java.nio.FloatBuffer;

// La interfaz que define el contrato nativo
public interface INativeManningSolver {

    // --- Constantes de Estrategia ---
    int MODE_SMART_LAZY = 0;    // Cálculo optimizado + Transferencia triangular (Default)
    int MODE_FULL_EVOLUTION = 1; // Cálculo robusto + Transferencia completa (Scientific)

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
            FloatBuffer initialDepths, // Estado base t=0
            FloatBuffer initialQ,      // Estado base t=0
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
     * @param mode          Estrategia de simulación (Use constantes MODE_*).
     * @param stride        Factor de submuestreo (1=Todos los pasos, N=Cada N pasos). Solo afecta a FULL_EVOLUTION.
     * @return Código de estado (0 = éxito).
     */
    int runBatch(
            long sessionHandle,
            FloatBuffer inputBuffer,
            FloatBuffer outputBuffer,
            int batchSize,
            int mode,
            int stride
    );

    // Lifecycle: Destroy
    void destroySession(long sessionHandle);
}