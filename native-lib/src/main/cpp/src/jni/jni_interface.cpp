#include <jni.h>
#include <vector>
#include <stdexcept>
#include <string>
#include <iostream>
#include "projectstalker/physics/manning_solver.h"
#include "projectstalker/physics/transport_solver.h"
// --- Declaraciones de Funciones JNI ---

extern "C" {

/*
 * Firma antigua para paso único (Mantenida por compatibilidad, lanza excepción).
 */
JNIEXPORT jfloatArray JNICALL
Java_projectstalker_physics_jni_NativeManningGpuSingleton_solveManningGpu(
    JNIEnv *env, jobject thiz,
    jfloatArray targetDischarges, jfloatArray initialDepthGuesses,
    jfloatArray bottomWidths, jfloatArray sideSlopes,
    jfloatArray manningCoefficients, jfloatArray bedSlopes
);

/*
 * NUEVA FIRMA OPTIMIZADA PARA BATCH
 * Acepta Buffers Directos (jobject) para la geometría estática.
 */
JNIEXPORT jfloatArray JNICALL
Java_projectstalker_physics_jni_NativeManningGpuSingleton_solveManningGpuBatch(
    JNIEnv *env, jobject thiz,
    jfloatArray gpuInitialGuess,    // Dinámico (Array)
    jfloatArray flatDischargeProfiles, // Dinámico (Array)
    jint batchSize,
    jint cellCount,
    jobject bottomWidthBuf,         // Estático (Direct Buffer)
    jobject sideSlopeBuf,           // Estático (Direct Buffer)
    jobject manningBuf,             // Estático (Direct Buffer)
    jobject bedSlopeBuf             // Estático (Direct Buffer)
);

JNIEXPORT jfloatArray JNICALL
Java_projectstalker_physics_jni_NativeTransportGpuSingleton_solveTransportEvolution(
    JNIEnv *env, jobject thiz,
    jobject cInBuf,      // DirectBuffer (Concentration)
    jobject uBuf,        // DirectBuffer (Velocity)
    jobject hBuf,        // DirectBuffer (Depth)
    jobject areaBuf,     // DirectBuffer (Area)
    jobject tempBuf,     // DirectBuffer (Temperature)
    jobject alphaBuf,    // DirectBuffer (Geometry Alpha)
    jobject decayBuf,    // DirectBuffer (Geometry Decay)
    jfloat dx,
    jfloat dtSub,
    jint numSteps,
    jint cellCount
);

} // extern "C"


// --- Funciones Auxiliares (Helpers) ---

/**
 * Helper: Lanza una RuntimeException en la JVM.
 */
static void throw_java_exception(JNIEnv *env, const char* message) {
    jclass exClass = env->FindClass("java/lang/RuntimeException");
    if (exClass != nullptr) {
        env->ThrowNew(exClass, message);
    }
}

/**
 * Helper: Convierte jfloatArray (Java Heap) a std::vector (C++ Heap).
 * Realiza una copia de memoria.
 */
static std::vector<float> jfloatarray_to_vector(JNIEnv *env, jfloatArray javaArray) {
    if (javaArray == nullptr) return {};

    jsize len = env->GetArrayLength(javaArray);
    jfloat *elements = env->GetFloatArrayElements(javaArray, 0);

    if (elements == nullptr) return {}; // OutOfMemory en JVM

    std::vector<float> vec(elements, elements + len);

    // JNI_ABORT libera el buffer sin copiar cambios de vuelta a Java (lectura solo)
    env->ReleaseFloatArrayElements(javaArray, elements, JNI_ABORT);

    return vec;
}

/**
 * Helper: Convierte std::vector a jfloatArray.
 */
static jfloatArray vector_to_jfloatarray(JNIEnv *env, const std::vector<float>& vec) {
    if (vec.empty()) return env->NewFloatArray(0);

    jfloatArray javaArray = env->NewFloatArray(static_cast<jsize>(vec.size()));
    if (javaArray == nullptr) return nullptr; // OutOfMemory

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
    throw_java_exception(env, "solveManningGpu (single step) no está optimizado ni implementado. Use solveBatch.");
    return nullptr;
}

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
    jobject bedSlopeBuf)
{
    try {
        // 1. OBTENER PUNTEROS DIRECTOS (Zero-Copy)
        // GetDirectBufferAddress devuelve la dirección de memoria raw del buffer.
        // Esto es extremadamente rápido y no duplica memoria.
        float* rawBottom  = (float*)env->GetDirectBufferAddress(bottomWidthBuf);
        float* rawSlope   = (float*)env->GetDirectBufferAddress(sideSlopeBuf);
        float* rawManning = (float*)env->GetDirectBufferAddress(manningBuf);
        float* rawBed     = (float*)env->GetDirectBufferAddress(bedSlopeBuf);

        // Validar que los buffers sean válidos (Direct Buffers correctos)
        if (!rawBottom || !rawSlope || !rawManning || !rawBed) {
            throw_java_exception(env, "Error JNI: Uno o más Geometry Buffers son nulos o no son DirectBuffers.");
            return nullptr;
        }

        // 2. PROCESAR DATOS DINÁMICOS (Arrays Normales)
        // Estos cambian en cada frame, así que seguimos copiándolos por ahora.
        std::vector<float> vInitialGuess = jfloatarray_to_vector(env, gpuInitialGuess);
        std::vector<float> vDischarges   = jfloatarray_to_vector(env, flatDischargeProfiles);

        // 3. ADAPTAR GEOMETRÍA A LA API DEL SOLVER (Puente Temporal)
        // Actualmente 'solve_manning_batch_cpp' espera std::vector.
        // Creamos vectores usando los punteros raw. Esto copia los datos al Heap de C++.
        // NOTA DE OPTIMIZACIÓN: En el futuro, cambiaremos solve_manning_batch_cpp para aceptar float* // y eliminaremos estas 4 líneas de copia.
        std::vector<float> vBottom(rawBottom, rawBottom + cellCount);
        std::vector<float> vSlope(rawSlope, rawSlope + cellCount);
        std::vector<float> vManning(rawManning, rawManning + cellCount);
        std::vector<float> vBed(rawBed, rawBed + cellCount);

        // 4. EJECUTAR SOLVER (C++ -> CUDA)
        std::vector<float> results = solve_manning_batch_cpp(
            vInitialGuess,
            vDischarges,
            (int)batchSize,
            (int)cellCount,
            vBottom,
            vSlope,
            vManning,
            vBed
        );

        // 5. RETORNAR RESULTADOS
        return vector_to_jfloatarray(env, results);

    } catch (const std::exception& e) {
        throw_java_exception(env, e.what());
        return nullptr;
    } catch (...) {
        throw_java_exception(env, "Error desconocido en capa nativa JNI.");
        return nullptr;
    }
}

JNIEXPORT jfloatArray JNICALL
Java_projectstalker_physics_jni_NativeTransportGpuSingleton_solveTransportEvolution(
    JNIEnv *env, jobject thiz,
    jobject cInBuf, jobject uBuf, jobject hBuf, jobject areaBuf,
    jobject tempBuf, jobject alphaBuf, jobject decayBuf,
    jfloat dx, jfloat dtSub, jint numSteps, jint cellCount)
{
    try {
        // Zero-Copy: Obtener punteros directos de los buffers de Java
        float* pC     = (float*)env->GetDirectBufferAddress(cInBuf);
        float* pU     = (float*)env->GetDirectBufferAddress(uBuf);
        float* pH     = (float*)env->GetDirectBufferAddress(hBuf);
        float* pA     = (float*)env->GetDirectBufferAddress(areaBuf);
        float* pT     = (float*)env->GetDirectBufferAddress(tempBuf);
        float* pAlpha = (float*)env->GetDirectBufferAddress(alphaBuf);
        float* pDecay = (float*)env->GetDirectBufferAddress(decayBuf);

        // Validación de seguridad
        if (!pC || !pU || !pH || !pA || !pT || !pAlpha || !pDecay) {
            throw_java_exception(env, "JNI Error (Transport): Uno de los buffers Directos es nulo o inválido.");
            return nullptr;
        }

        // Llamada al orquestador C++
        std::vector<float> result = solve_transport_evolution_cpp(
            pC, pU, pH, pA, pT, pAlpha, pDecay,
            dx, dtSub, numSteps, cellCount
        );

        return vector_to_jfloatarray(env, result);

    } catch (const std::exception& e) {
        throw_java_exception(env, e.what());
        return nullptr;
    } catch (...) {
        throw_java_exception(env, "Error desconocido en capa nativa JNI (Transport).");
        return nullptr;
    }
}