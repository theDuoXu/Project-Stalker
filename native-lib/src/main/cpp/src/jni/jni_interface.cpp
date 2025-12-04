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
// SECCIÓN 1: NUEVA API MANNING (Stateful + Zero Copy Híbrido + Flyweight)
// =============================================================================

/*
 * Inicializa la sesión GPU.
 * Usa DirectBuffers para la geometría estática y ahora también para el ESTADO INICIAL (Flyweight).
 */
JNIEXPORT jlong JNICALL
Java_projectstalker_physics_jni_NativeManningGpuSingleton_initSession(
    JNIEnv *env, jobject thiz,
    jobject widthBuf, jobject slopeBuf, jobject manningBuf, jobject bedBuf,
    jobject initDepthBuf, jobject initQBuf, // <--- NUEVOS ARGUMENTOS
    jint cellCount
) {
    try {
        // Obtener punteros directos (Zero-Copy desde DirectBuffers)
        float* pWidth   = (float*)env->GetDirectBufferAddress(widthBuf);
        float* pSlope   = (float*)env->GetDirectBufferAddress(slopeBuf);
        float* pManning = (float*)env->GetDirectBufferAddress(manningBuf);
        float* pBed     = (float*)env->GetDirectBufferAddress(bedBuf);

        // Obtener punteros del estado inicial (Solo se leen una vez)
        float* pInitDepth = (float*)env->GetDirectBufferAddress(initDepthBuf);
        float* pInitQ     = (float*)env->GetDirectBufferAddress(initQBuf);

        // Validación básica de punteros
        if (!pWidth || !pSlope || !pManning || !pBed || !pInitDepth || !pInitQ) {
            throw_java_exception(env, "Error JNI Init: Buffers inválidos (deben ser DirectBuffers).");
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
 * Ejecuta el batch.
 * MODIFICADO: Ahora es mucho más ligero (Flyweight).
 * Solo recibe 'newInflowsArr'. El estado inicial ya reside en la GPU.
 */
JNIEXPORT jfloatArray JNICALL
Java_projectstalker_physics_jni_NativeManningGpuSingleton_runBatch(
    JNIEnv *env, jobject thiz,
    jlong handle,
    jfloatArray newInflowsArr
    // ELIMINADOS: initialDepthsArr, initialQArr (Ya están en la sesión)
) {
    // 1. Validar Sesión
    ManningSession* session = (ManningSession*)handle;
    if (!session) {
        throw_java_exception(env, "Error JNI Run: La sesión de Manning es nula o ha sido cerrada.");
        return nullptr;
    }

    // 2. Acceso Crítico a Arrays Java (Pinning)
    // Solo necesitamos pinear el array de inflows (que es pequeño)
    void* pNewInflows = env->GetPrimitiveArrayCritical(newInflowsArr, nullptr);

    if (!pNewInflows) {
        throw_java_exception(env, "Error JNI Run: Fallo crítico al acceder a arrays de memoria.");
        return nullptr;
    }

    jsize batchSize = env->GetArrayLength(newInflowsArr);
    std::vector<float> results;
    std::string errorMsg;

    // 3. Ejecución C++ (Protegida)
    try {
        results = run_manning_batch_stateful(
            session,
            (float*)pNewInflows,
            (int)batchSize
        );
    } catch (const std::exception& e) {
        errorMsg = e.what();
    } catch (...) {
        errorMsg = "Error desconocido nativo en Manning Run.";
    }

    // 4. Liberar Pinning (JNI_ABORT = No copiar de vuelta, solo leímos inputs)
    env->ReleasePrimitiveArrayCritical(newInflowsArr, pNewInflows, JNI_ABORT);

    // 5. Manejo de Errores Diferido
    if (!errorMsg.empty()) {
        throw_java_exception(env, errorMsg.c_str());
        return nullptr;
    }

    // 6. Convertir y retornar resultados
    // Nota: El vector 'results' ahora es más pequeño (Triangular/Cuadrado), Java deberá reconstruirlo.
    return vector_to_jfloatarray(env, results);
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
    throw_java_exception(env, "MÉTODO OBSOLETO: solveManningGpu. Use la nueva API Stateful (initSession/runBatch).");
    return nullptr;
}

JNIEXPORT jfloatArray JNICALL
Java_projectstalker_physics_jni_NativeManningGpuSingleton_solveManningGpuBatch(
    JNIEnv *env, jobject thiz,
    jfloatArray gpuInitialGuess, jfloatArray flatDischargeProfiles,
    jint batchSize, jint cellCount,
    jobject bottomWidthBuf, jobject sideSlopeBuf, jobject manningBuf, jobject bedSlopeBuf)
{
    throw_java_exception(env, "MÉTODO OBSOLETO: solveManningGpuBatch. Use la nueva API Stateful (initSession/runBatch).");
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

} // extern "C"