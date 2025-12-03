package projectstalker.physics.jni;

import java.nio.FloatBuffer;

// La interfaz que define el contrato nativo
public interface INativeManningSolver {

    // Lifecycle: Init (Geometría Estática -> Buffers Directos)
    long initSession(FloatBuffer bottomWidths, FloatBuffer sideSlopes, FloatBuffer manningCoefficients, FloatBuffer bedSlopes, int cellCount);

    // Lifecycle: Run (Datos Dinámicos -> Arrays Primitivos para Pinning rápido)
    // Nota: newInflows es el array comprimido [BatchSize]
    float[] runBatch(long sessionHandle, float[] newInflows, float[] initialDepths, float[] initialQ);

    // Lifecycle: Destroy
    void destroySession(long sessionHandle);
}