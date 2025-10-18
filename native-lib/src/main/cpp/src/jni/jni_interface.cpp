// jni_interface.cpp
#include <jni.h>
#include <vector>
#include <stdexcept>
#include "projectstalker/physics/manning_solver.h" // Nuestro orquestador C++ (Paso 2)

// --- Declaraciones de Funciones JNI ---

// Declaramos las dos funciones nativas que implementaremos
extern "C" {

JNIEXPORT jfloatArray JNICALL
Java_projectstalker_physics_jni_NativeManningGpuSingleton_solveManningGpu(
    JNIEnv *env, jobject thiz,
    jfloatArray targetDischarges, jfloatArray initialDepthGuesses,
    jfloatArray bottomWidths, jfloatArray sideSlopes,
    jfloatArray manningCoefficients, jfloatArray bedSlopes
);

JNIEXPORT jfloatArray JNICALL
Java_projectstalker_physics_jni_NativeManningGpuSingleton_solveManningGpuBatch(
    JNIEnv *env, jobject thiz,
    jfloatArray gpuInitialGuess, jfloatArray flatDischargeProfiles,
    jint batchSize, jint cellCount,
    jfloatArray bottomWidths, jfloatArray sideSlopes,
    jfloatArray manningCoefficients, jfloatArray bedSlopes
);

} // extern "C"


// --- Funciones Auxiliares (Helpers) ---

/**
 * Función auxiliar para convertir un jfloatArray de JNI a un std::vector<float> de C++.
 * Maneja la obtención y liberación de la memoria de Java.
 */
static std::vector<float> jfloatarray_to_vector(JNIEnv *env, jfloatArray javaArray) {
    if (javaArray == nullptr) {
        return {};
    }
    jsize len = env->GetArrayLength(javaArray);
    jfloat *elements = env->GetFloatArrayElements(javaArray, 0);
    if (elements == nullptr) {
        // Falló la obtención de memoria
        return {};
    }
    // Copiar los datos en un vector de C++
    std::vector<float> vec(elements, elements + len);

    // Liberar la memoria de Java (modo 0 = copiar de vuelta y liberar)
    env->ReleaseFloatArrayElements(javaArray, elements, 0);
    return vec;
}

/**
 * Función auxiliar para convertir un std::vector<float> de C++ a un jfloatArray de JNI.
 */
static jfloatArray vector_to_jfloatarray(JNIEnv *env, const std::vector<float>& vec) {
    if (vec.empty()) {
        return env->NewFloatArray(0);
    }
    jfloatArray javaArray = env->NewFloatArray(vec.size());
    if (javaArray == nullptr) {
        // Falló la creación del array (ej. OutOfMemory)
        return nullptr;
    }
    // Copiar los datos del vector al nuevo array de Java
    env->SetFloatArrayRegion(javaArray, 0, vec.size(), vec.data());
    return javaArray;
}

/**
 * Lanza una excepción en Java.
 */
static void throw_java_exception(JNIEnv *env, const char* message) {
    // Busca la clase RuntimeException (o una más específica si se desea)
    jclass exClass = env->FindClass("java/lang/RuntimeException");
    if (exClass != nullptr) {
        env->ThrowNew(exClass, message);
    }
}

// --- Implementación JNI: solveManningGpu (Paso único) ---

JNIEXPORT jfloatArray JNICALL
Java_projectstalker_physics_jni_NativeManningGpuSingleton_solveManningGpu(
    JNIEnv *env, jobject thiz,
    jfloatArray targetDischarges, jfloatArray initialDepthGuesses,
    jfloatArray bottomWidths, jfloatArray sideSlopes,
    jfloatArray manningCoefficients, jfloatArray bedSlopes)
{
    // Esta función no está implementada en el lado C++ (solve_manning_single_cpp)
    // en nuestro plan actual, pero si lo estuviera, la lógica sería idéntica
    // a la de 'solveManningGpuBatch'.
    // Por ahora, lanzamos una excepción para que Java sepa que no está implementada.
    throw_java_exception(env, "solveManningGpu (paso único) no está implementado en la capa nativa.");
    return nullptr;
}

// --- Implementación JNI: solveManningGpuBatch (Lote) ---

JNIEXPORT jfloatArray JNICALL
Java_projectstalker_physics_jni_NativeManningGpuSingleton_solveManningGpuBatch(
    JNIEnv *env, jobject thiz,
    jfloatArray gpuInitialGuess, jfloatArray flatDischargeProfiles,
    jint batchSize, jint cellCount,
    jfloatArray bottomWidths, jfloatArray sideSlopes,
    jfloatArray manningCoefficients, jfloatArray bedSlopes)
{
    try {
        // 1. Convertir todos los arrays de Java a vectores de C++
        std::vector<float> initialGuess_vec = jfloatarray_to_vector(env, gpuInitialGuess);
        std::vector<float> discharges_vec = jfloatarray_to_vector(env, flatDischargeProfiles);
        std::vector<float> widths_vec = jfloatarray_to_vector(env, bottomWidths);
        std::vector<float> slopes_vec = jfloatarray_to_vector(env, sideSlopes);
        std::vector<float> manning_vec = jfloatarray_to_vector(env, manningCoefficients);
        std::vector<float> bedSlopes_vec = jfloatarray_to_vector(env, bedSlopes);

        // 2. Llamar a la función del orquestador C++ (Paso 2)
        std::vector<float> results_vec = solve_manning_batch_cpp(
            initialGuess_vec,
            discharges_vec,
            (int)batchSize,
            (int)cellCount,
            widths_vec,
            slopes_vec,
            manning_vec,
            bedSlopes_vec
        );

        // 3. Convertir el vector de resultado de C++ a un jfloatArray de Java
        return vector_to_jfloatarray(env, results_vec);

    } catch (const std::exception& e) {
        // Si C++ lanzó una excepción (ej. error de CUDA), la atrapamos
        // y la volvemos a lanzar como una excepción de Java.
        throw_java_exception(env, e.what());
        return nullptr;
    }
}