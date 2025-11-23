#include <jni.h>
#include <vector>
#include <stdexcept>
#include <string>
#include <iostream>
#include "projectstalker/physics/manning_solver.h"
#include "projectstalker/physics/transport_solver.h"

// --- Declaraciones de Funciones JNI ---

extern "C" {

// (Antigua firma single-step, se mantiene igual)
JNIEXPORT jfloatArray JNICALL
Java_projectstalker_physics_jni_NativeManningGpuSingleton_solveManningGpu(
    JNIEnv *env, jobject thiz,
    jfloatArray targetDischarges, jfloatArray initialDepthGuesses,
    jfloatArray bottomWidths, jfloatArray sideSlopes,
    jfloatArray manningCoefficients, jfloatArray bedSlopes
);

// Firma Batch Manning
JNIEXPORT jfloatArray JNICALL
Java_projectstalker_physics_jni_NativeManningGpuSingleton_solveManningGpuBatch(
    JNIEnv *env, jobject thiz,
    jfloatArray gpuInitialGuess,
    jfloatArray flatDischargeProfiles,
    jint batchSize,
    jint cellCount,
    jobject bottomWidthBuf,
    jobject sideSlopeBuf,
    jobject manningBuf,
    jobject bedSlopeBuf
);

// Firma Transporte
JNIEXPORT jfloatArray JNICALL
Java_projectstalker_physics_jni_NativeTransportGpuSingleton_solveTransportEvolution(
    JNIEnv *env, jobject thiz,
    jobject cInBuf, jobject uBuf, jobject hBuf, jobject areaBuf,
    jobject tempBuf, jobject alphaBuf, jobject decayBuf,
    jfloat dx, jfloat dtSub, jint numSteps, jint cellCount
);

} // extern "C"


// --- Funciones Auxiliares (Helpers) ---

static void throw_java_exception(JNIEnv *env, const char* message) {
    jclass exClass = env->FindClass("java/lang/RuntimeException");
    if (exClass != nullptr) {
        env->ThrowNew(exClass, message);
    }
}

// (Helper jfloatarray_to_vector eliminado porque ya no lo usamos para inputs!)
// Mantenemos vector_to_jfloatarray para el output.
static jfloatArray vector_to_jfloatarray(JNIEnv *env, const std::vector<float>& vec) {
    if (vec.empty()) return env->NewFloatArray(0);
    jfloatArray javaArray = env->NewFloatArray(static_cast<jsize>(vec.size()));
    if (javaArray == nullptr) return nullptr;
    env->SetFloatArrayRegion(javaArray, 0, static_cast<jsize>(vec.size()), vec.data());
    return javaArray;
}


// --- Implementaciones ---

JNIEXPORT jfloatArray JNICALL
Java_projectstalker_physics_jni_NativeManningGpuSingleton_solveManningGpu(
    JNIEnv *env, jobject thiz,
    jfloatArray targetDischarges, jfloatArray initialDepthGuesses,
    jfloatArray bottomWidths, jfloatArray sideSlopes,
    jfloatArray manningCoefficients, jfloatArray bedSlopes)
{
    throw_java_exception(env, "solveManningGpu (single step) obsoleto. Use solveBatch.");
    return nullptr;
}

JNIEXPORT jfloatArray JNICALL
Java_projectstalker_physics_jni_NativeManningGpuSingleton_solveManningGpuBatch(
    JNIEnv *env, jobject thiz,
    jfloatArray gpuInitialGuess, jfloatArray flatDischargeProfiles,
    jint batchSize, jint cellCount,
    jobject bottomWidthBuf, jobject sideSlopeBuf, jobject manningBuf, jobject bedSlopeBuf)
{
    // MANNING: DATOS DINÁMICOS (Arrays) -> Pinning
    jfloat* initialGuess = nullptr;
    jfloat* discharges = nullptr;

    try {
        // 1. OBTENER PUNTEROS DIRECTOS (Zero-Copy) para geometría
        float* rawBottom  = (float*)env->GetDirectBufferAddress(bottomWidthBuf);
        float* rawSlope   = (float*)env->GetDirectBufferAddress(sideSlopeBuf);
        float* rawManning = (float*)env->GetDirectBufferAddress(manningBuf);
        float* rawBed     = (float*)env->GetDirectBufferAddress(bedSlopeBuf);

        if (!rawBottom || !rawSlope || !rawManning || !rawBed) {
            throw_java_exception(env, "Error JNI Manning: Buffers de geometría inválidos.");
            return nullptr;
        }

        // 2. OBTENER PUNTEROS DE ARRAYS DINÁMICOS (GetPrimitiveArrayCritical es más rápido)
        // Usamos GetFloatArrayElements estándar por seguridad
        initialGuess = env->GetFloatArrayElements(gpuInitialGuess, NULL);
        discharges   = env->GetFloatArrayElements(flatDischargeProfiles, NULL);

        if (!initialGuess || !discharges) {
            // Si falla (OutOfMemory), liberar lo que se haya pillado
            if (initialGuess) env->ReleaseFloatArrayElements(gpuInitialGuess, initialGuess, JNI_ABORT);
            throw_java_exception(env, "Error JNI Manning: Fallo al obtener arrays dinámicos.");
            return nullptr;
        }

        // 3. LLAMADA DIRECTA (Sin copias a std::vector)
        std::vector<float> results = solve_manning_batch_cpp(
            initialGuess, discharges,
            (int)batchSize, (int)cellCount,
            rawBottom, rawSlope, rawManning, rawBed
        );

        // 4. Liberar arrays Java (JNI_ABORT porque solo leímos)
        env->ReleaseFloatArrayElements(gpuInitialGuess, initialGuess, JNI_ABORT);
        env->ReleaseFloatArrayElements(flatDischargeProfiles, discharges, JNI_ABORT);

        return vector_to_jfloatarray(env, results);

    } catch (const std::exception& e) {
        // Limpieza segura en caso de excepción C++
        if (initialGuess) env->ReleaseFloatArrayElements(gpuInitialGuess, initialGuess, JNI_ABORT);
        if (discharges) env->ReleaseFloatArrayElements(flatDischargeProfiles, discharges, JNI_ABORT);

        throw_java_exception(env, e.what());
        return nullptr;
    } catch (...) {
        if (initialGuess) env->ReleaseFloatArrayElements(gpuInitialGuess, initialGuess, JNI_ABORT);
        if (discharges) env->ReleaseFloatArrayElements(flatDischargeProfiles, discharges, JNI_ABORT);

        throw_java_exception(env, "Error desconocido en capa nativa Manning.");
        return nullptr;
    }
}

// --- Implementación Transporte (Zero-Copy Total) ---
// Se mantiene igual que la versión anterior, ya era óptima.
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