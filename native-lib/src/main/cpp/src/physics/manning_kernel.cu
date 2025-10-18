// manning_kernel.cu
#include "projectstalker/physics/manning_kernel.h"
#include <cuda_runtime.h>
#include <cmath> // para fmaxf, sqrtf, powf, copysignf

// --- Constantes del Kernel ---

// Número fijo de iteraciones de Newton-Raphson.
// Se ejecuta siempre este número de veces, sin 'if' de convergencia.
#define MAX_ITERATIONS 20

// Épsilon para la única comprobación de división por cero necesaria (dQ/dH).
#define SAFE_EPSILON 1e-7f

// Mínima profundidad permitida para evitar singularidades (coincide con Java).
#define MIN_DEPTH 0.001f

// --- Funciones de Dispositivo (Helpers en la GPU) ---

__device__ inline float device_calculateA(float H, float b, float m) {
    return (b + m * H) * H;
}

__device__ inline float device_calculateP(float H, float b, float m) {
    return b + 2.0f * H * sqrtf(1.0f + m * m);
}

/**
 * __device__: Calcula el Caudal (Q).
 * No necesita comprobaciones de división por cero gracias a la
 * sanitización de H > 0 en Java, que garantiza A > 0 y P > 0.
 */
__device__ inline float device_calculateQ(float H, float b, float m, float n, float S) {
    float A = device_calculateA(H, b, m);
    float P = device_calculateP(H, b, m);
    float R = A / P;

    return (1.0f / n) * A * powf(R, 2.0f / 3.0f) * sqrtf(S);
}

/**
 * __device__: Calcula la derivada dQ/dH.
 * No necesita comprobaciones en A y P gracias a la sanitización de H.
 */
__device__ inline float device_calculate_dQ_dH(float H, float b, float m, float currentQ) {
    // Re-calculamos A y P para dQ/dH
    float A = device_calculateA(H, b, m);
    float P = device_calculateP(H, b, m);

    float term_A_derivative = (5.0f * (b + 2.0f * m * H)) / A;
    float term_P_derivative = (4.0f * sqrtf(1.0f + m * m)) / P;

    return (currentQ / 3.0f) * (term_A_derivative - term_P_derivative);
}

// --- Kernel Principal ---

/**
 * __global__: El kernel de CUDA. Cada hilo ejecuta esta función.
 */
__global__ void manningSolverKernel(
    float* d_results,
    const float* d_initialDepths,
    const float* d_targetDischarges,
    const float* d_bottomWidths,
    const float* d_sideSlopes,
    const float* d_manningCoeffs,
    const float* d_bedSlopes,
    int totalThreads,
    int cellCount
) {
    // 1. Calcular el ID global del hilo
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // 2. Límite del Grid (El único 'if' de control permitido)
    if (id >= totalThreads) {
        return;
    }

    // 3. Calcular el índice de geometría
    // El 'id' global es [t0_c0, t0_c1, ..., t1_c0, ...].
    // El 'geo_id' es [c0, c1, ..., c0, ...].
    int geo_id = id % cellCount;

    // 4. Cargar datos desde la memoria Global a registros
    float H = d_initialDepths[id]; // Sanitizado en Java (> 0)
    const float Q_target = d_targetDischarges[id]; // Sanitizado en Java (> 0)
    const float b = d_bottomWidths[geo_id];
    const float m = d_sideSlopes[geo_id];
    const float n = d_manningCoeffs[geo_id];
    const float S = d_bedSlopes[geo_id]; // Sanitizado en Java (> 0)

    // 5. Bucle Newton-Raphson (Iteración Fija, sin 'if' de convergencia)
    #pragma unroll
    for (int i = 0; i < MAX_ITERATIONS; i++) {

        float Q_calc = device_calculateQ(H, b, m, n, S);
        float f_H = Q_calc - Q_target;

        // Pasamos 'b' y 'm' en lugar de 'n' y 'S' que no son necesarios
        float dQ_dH = device_calculate_dQ_dH(H, b, m, Q_calc);

        // **ÚNICA COMPROBACIÓN CRÍTICA (SIN RAMAS)**
        // Previene la división por cero si la derivada es plana (dQ_dH = 0).
        // copysignf transfiere el signo de dQ_dH a SAFE_EPSILON.
        float dQ_dH_safe = dQ_dH + copysignf(SAFE_EPSILON, dQ_dH);

        float H_next = H - f_H / dQ_dH_safe;

        // Saneamiento SIN RAMAS: H no puede ser negativo.
        H = fmaxf(MIN_DEPTH, H_next);
    }

    // 6. Cálculo Final de Velocidad (después del bucle)
    // A_final está garantizado > 0 porque H >= MIN_DEPTH.
    float A_final = device_calculateA(H, b, m);
    float V = Q_target / A_final;

    // 7. Escribir resultados intercalados en memoria Global
    int result_idx = id * 2;
    d_results[result_idx] = H;
    d_results[result_idx + 1] = V;
}


// --- Función Lanzadora (Launcher) ---

/**
 * Función C++ (Host) que configura y lanza el kernel de CUDA.
 */
void launchManningKernel(
    float* d_results,
    const float* d_initialDepths,
    const float* d_targetDischarges,
    const float* d_bottomWidths,
    const float* d_sideSlopes,
    const float* d_manningCoeffs,
    const float* d_bedSlopes,
    int batchSize,
    int cellCount
) {
    int totalThreads = batchSize * cellCount;
    if (totalThreads == 0) return;

    // Usamos bloques de 256 hilos (un buen valor estándar).
    const int threadsPerBlock = 256;

    // Calcula el número de bloques necesarios (división entera redondeando hacia arriba).
    const int blocksPerGrid = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

    // Lanzar el kernel
    manningSolverKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_results,
        d_initialDepths,
        d_targetDischarges,
        d_bottomWidths,
        d_sideSlopes,
        d_manningCoeffs,
        d_bedSlopes,
        totalThreads,
        cellCount
    );
}