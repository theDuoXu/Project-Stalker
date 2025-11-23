package projectstalker.physics.jni;

import java.nio.FloatBuffer;

/**
 * Interfaz para desacoplar la implementación nativa JNI del resto de la aplicación.
 * Permite usar Mocks en tests unitarios sin cargar la librería CUDA.
 */
public interface INativeTransportSolver {

    /**
     * Llama al código nativo para evolucionar el sistema de transporte.
     * Todos los buffers deben ser Direct Buffers (allocateDirect).
     */
    float[] solveTransportEvolution(
            FloatBuffer cInBuf,      // Concentración
            FloatBuffer uBuf,        // Velocidad
            FloatBuffer hBuf,        // Profundidad
            FloatBuffer areaBuf,     // Área
            FloatBuffer tempBuf,     // Temperatura
            FloatBuffer alphaBuf,    // Alpha (Geometría)
            FloatBuffer decayBuf,    // Decay (Geometría)
            float dx,
            float dtSub,
            int numSteps,
            int cellCount
    );
}