// src/main/cpp/src/jni/jni_interface.cpp
#include <jni.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>

// Incluimos ambos headers: el nuevo refactorizado y el viejo de transporte
#include "projectstalker/physics/manning_solver.h"
#include "projectstalker/physics/transport_solver.h"

// --- Helpers JNI ---

// Lanza una RuntimeException en Java de forma segura
static void throw_java_exception(JNIEnv *env, const char* message) {
    jclass exClass = env->FindClass("java/lang/RuntimeException");
    if (exClass != nullptr) {
        env->ThrowNew(exClass, message);
    }
}

// Convierte std::vector<float> a jfloatArray para devolver resultados
// NOTA: Se mantiene para la API de Transporte (Legacy) que aún usa vectores.
static jfloatArray vector_to_jfloatarray(JNIEnv *env, const std::vector<float>& vec) {
    if (vec.empty()) return env->NewFloatArray(0);

    // Check de seguridad por si el vector es monstruoso
    if (vec.size() > (size_t)2147483647) {
        throw_java_exception(env, "El resultado de la GPU excede el tamaño máximo de array en Java.");
        return nullptr;
    }

    jsize size = static_cast<jsize>(vec.size());
    jfloatArray result = env->NewFloatArray(size);
    if (result == nullptr) return nullptr; // OutOfMemoryError ya lanzado por JVM

    env->SetFloatArrayRegion(result, 0, size, vec.data());
    return result;
}

extern "C" {

// =============================================================================
// SECCIÓN 1: NUEVA API MANNING (Stateful + Zero Copy Total + Flyweight)
// =============================================================================

/*
 * Inicializa la sesión GPU.
 * Usa DirectBuffers para la geometría estática y el ESTADO INICIAL (Flyweight).
 * Mantiene la lógica de "Punteros Directos" ya implementada en el paso anterior.
 */
JNIEXPORT jlong JNICALL
Java_projectstalker_physics_jni_NativeManningGpuSingleton_initSession(
    JNIEnv *env, jobject thiz,
    jobject widthBuf, jobject slopeBuf, jobject manningBuf, jobject bedBuf,
    jobject initDepthBuf, jobject initQBuf,
    jint cellCount
) {
    try {
        // Obtener punteros directos (Zero-Copy desde DirectBuffers)
        // Java garantiza que la memoria está "Pinned" al usar allocateDirect()
        float* pWidth   = (float*)env->GetDirectBufferAddress(widthBuf);
        float* pSlope   = (float*)env->GetDirectBufferAddress(slopeBuf);
        float* pManning = (float*)env->GetDirectBufferAddress(manningBuf);
        float* pBed     = (float*)env->GetDirectBufferAddress(bedBuf);

        // Obtener punteros del estado inicial (Solo se leen una vez)
        float* pInitDepth = (float*)env->GetDirectBufferAddress(initDepthBuf);
        float* pInitQ     = (float*)env->GetDirectBufferAddress(initQBuf);

        // Validación básica de punteros
        if (!pWidth || !pSlope || !pManning || !pBed || !pInitDepth || !pInitQ) {
            throw_java_exception(env, "Error JNI Init: Buffers inválidos. Asegúrese de usar ByteBuffer.allocateDirect().");
            return 0;
        }

        // Llamada al constructor C++ (Baking + Carga de Estado Base)
        ManningSession* session = init_manning_session(
            pWidth, pSlope, pManning, pBed,
            pInitDepth, pInitQ,
            (int)cellCount
        );

        // Retornar puntero opaco
        return (jlong)session;

    } catch (const std::exception& e) {
        throw_java_exception(env, e.what());
        return 0;
    }
}

/*
 * Ejecuta el batch usando DMA (Zero-Copy).
 * MODIFICADO:
 * 1. Recibe Buffers de Entrada y Salida pre-asignados (Input/Output).
 * 2. Recibe 'mode' para seleccionar estrategia (Smart vs Full).
 * 3. Recibe 'stride' para muestreo de salida (Nuevo).
 * 4. No devuelve array (escribe in-place).
 */
JNIEXPORT jint JNICALL
Java_projectstalker_physics_jni_NativeManningGpuSingleton_runBatch(
    JNIEnv *env, jobject thiz,
    jlong handle,
    jobject inputBuf,  // Buffer Directo de Entrada (Inflows)
    jobject outputBuf, // Buffer Directo de Salida (Results)
    jint batchSize,
    jint mode,         // 0=Smart, 1=Full
    jint stride
) {
    // 1. Validar Sesión
    ManningSession* session = (ManningSession*)handle;
    if (!session) {
        throw_java_exception(env, "Error JNI Run: La sesión de Manning es nula o ha sido cerrada.");
        return -1;
    }

    // 2. Obtener Direcciones de Memoria Pinned (Zero-Copy)
    // GetDirectBufferAddress devuelve el puntero crudo a la memoria física.
    // Coste: Casi cero.
    float* pInput  = (float*)env->GetDirectBufferAddress(inputBuf);
    float* pOutput = (float*)env->GetDirectBufferAddress(outputBuf);

    if (!pInput || !pOutput) {
        throw_java_exception(env, "Error JNI Run: Buffers no son directos (allocateDirect) o son inválidos.");
        return -2;
    }

    // 3. Ejecución C++ (Protegida)
    // Los datos viajan GPU <-> RAM(Pinned) directamente vía DMA.
    try {
        run_manning_batch_stateful(
            session,
            pInput,
            pOutput,
            (int)batchSize,
            (int)mode,
            (int)stride // <--- Pasamos el stride
        );
        return 0; // Éxito

    } catch (const std::exception& e) {
        throw_java_exception(env, e.what());
        return -3;
    } catch (...) {
        throw_java_exception(env, "Error desconocido nativo en Manning Run.");
        return -4;
    }
}

/*
 * Destruye la sesión y libera VRAM.
 */
JNIEXPORT void JNICALL
Java_projectstalker_physics_jni_NativeManningGpuSingleton_destroySession(
    JNIEnv *env, jobject thiz,
    jlong handle
) {
    ManningSession* session = (ManningSession*)handle;
    if (session) {
        destroy_manning_session(session);
    }
}


// =============================================================================
// SECCIÓN 2: API LEGACY (Deprecated / Disabled)
// =============================================================================

JNIEXPORT jfloatArray JNICALL
Java_projectstalker_physics_jni_NativeManningGpuSingleton_solveManningGpu(
    JNIEnv *env, jobject thiz,
    jfloatArray targetDischarges, jfloatArray initialDepthGuesses,
    jfloatArray bottomWidths, jfloatArray sideSlopes,
    jfloatArray manningCoefficients, jfloatArray bedSlopes)
{
    throw_java_exception(env, "MÉTODO OBSOLETO: solveManningGpu. Use la nueva API Stateful DMA.");
    return nullptr;
}

JNIEXPORT jfloatArray JNICALL
Java_projectstalker_physics_jni_NativeManningGpuSingleton_solveManningGpuBatch(
    JNIEnv *env, jobject thiz,
    jfloatArray gpuInitialGuess, jfloatArray flatDischargeProfiles,
    jint batchSize, jint cellCount,
    jobject bottomWidthBuf, jobject sideSlopeBuf, jobject manningBuf, jobject bedSlopeBuf)
{
    throw_java_exception(env, "MÉTODO OBSOLETO: solveManningGpuBatch. Use la nueva API Stateful DMA.");
    return nullptr;
}


// =============================================================================
// SECCIÓN 3: API TRANSPORTE
// =============================================================================

JNIEXPORT jfloatArray JNICALL
Java_projectstalker_physics_jni_NativeTransportGpuSingleton_solveTransportEvolution(
    JNIEnv *env, jobject thiz,
    jobject cInBuf, jobject uBuf, jobject hBuf, jobject areaBuf,
    jobject tempBuf, jobject alphaBuf, jobject decayBuf,
    jfloat dx, jfloat dtSub, jint numSteps, jint cellCount)
{
    try {
        float* pC     = (float*)env->GetDirectBufferAddress(cInBuf);
        float* pU     = (float*)env->GetDirectBufferAddress(uBuf);
        float* pH     = (float*)env->GetDirectBufferAddress(hBuf);
        float* pA     = (float*)env->GetDirectBufferAddress(areaBuf);
        float* pT     = (float*)env->GetDirectBufferAddress(tempBuf);
        float* pAlpha = (float*)env->GetDirectBufferAddress(alphaBuf);
        float* pDecay = (float*)env->GetDirectBufferAddress(decayBuf);

        if (!pC || !pU || !pH || !pA || !pT || !pAlpha || !pDecay) {
            throw_java_exception(env, "JNI Error Transport: Buffers inválidos.");
            return nullptr;
        }

        std::vector<float> result = solve_transport_evolution_cpp(
            pC, pU, pH, pA, pT, pAlpha, pDecay,
            dx, dtSub, numSteps, cellCount
        );

        return vector_to_jfloatarray(env, result);

    } catch (const std::exception& e) {
        throw_java_exception(env, e.what());
        return nullptr;
    } catch (...) {
        throw_java_exception(env, "Error desconocido en Transporte.");
        return nullptr;
    }
}

// =============================================================================
// SECCIÓN 4: FUNCIONALIDADES DE VERIFICACIÓN
// =============================================================================
/*
 * Class:     projectstalker_physics_jni_NativeManningGpuSingleton
 * Method:    getDeviceCount
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_projectstalker_physics_jni_NativeManningGpuSingleton_getDeviceCount
  (JNIEnv *env, jobject obj) {

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(error_id));
        return 0;
    }

    //Intentar obtener propiedades del dispositivo 0 para asegurar que está vivo
    if (deviceCount > 0) {
        cudaDeviceProp deviceProp;
        if (cudaGetDeviceProperties(&deviceProp, 0) != cudaSuccess) {
            return 0;
        }
        // Imprimir nombre de la gráfica en logs nativos
        printf("Detectada GPU: %s\n", deviceProp.name);
    }

    return deviceCount;
}

} // extern "C"