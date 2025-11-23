// src/main/cpp/src/physics/manning_kernel.cu
#include "projectstalker/physics/manning_kernel.h"
#include <cuda_runtime.h>
#include <cmath>

// --- Constantes ---
#define MAX_ITERATIONS 20
#define SAFE_EPSILON 1e-7f
#define MIN_DEPTH 0.001f

// Constantes pre-calculadas para evitar divisiones en tiempo de ejecución
#define ONE_THIRD 0.33333333f
#define FIVE_THIRDS 1.66666667f
#define FOUR_THIRDS 1.33333333f
#define TWO_THIRDS 0.66666667f

// --- Helpers ---

__device__ inline float device_calculateA(float H, float b, float m) {
    return (b + m * H) * H;
}

// Recibe pythagoras = sqrt(1 + m^2) pre-calculado
__device__ inline float device_calculateP_optimized(float H, float b, float pythagoras) {
    return b + 2.0f * H * pythagoras;
}

__device__ inline float device_calculateTopWidth(float H, float b, float m) {
    return b + 2.0f * m * H;
}

/**
 * Calcula Q.
 * R^(2/3) es la operación costosa (SFU).
 */
__device__ inline float device_calculateQ_opt(float A, float P, float inv_n, float sqrt_slope) {
    // Evitamos división por cero con fmaxf en P (aunque H saneado lo evita, es un seguro barato)
    float R = A / fmaxf(P, SAFE_EPSILON);
    return inv_n * A * powf(R, TWO_THIRDS) * sqrt_slope;
}

/**
 * Calcula dQ/dH factorizado.
 * Derivada analítica optimizada para reducir instrucciones.
 */
__device__ inline float device_calculate_dQ_dH_opt(
    float A, float P, float T, float pythagoras, float Q
) {
    // dQ/dH = Q * [ (5/3)*(T/A) - (2/3)*(dP/dH)/P ]
    // dP/dH = 2 * pythagoras
    // Termino P = (2/3) * (2 * pyth) / P = (4/3) * pyth / P

    float term_A = (FIVE_THIRDS * T) / fmaxf(A, SAFE_EPSILON);
    float term_P = (FOUR_THIRDS * pythagoras) / fmaxf(P, SAFE_EPSILON);

    return Q * (term_A - term_P);
}

// --- Kernel Principal ---

__global__ void manningSolverKernel(
    float* __restrict__ d_results, // [H, V, H, V...]
    const float* __restrict__ d_initialDepths,
    const float* __restrict__ d_targetDischarges,
    const float* __restrict__ d_bottomWidths,
    const float* __restrict__ d_sideSlopes,
    const float* __restrict__ d_manningCoeffs,
    const float* __restrict__ d_bedSlopes,
    int totalThreads,
    int cellCount
) {
    // 1. ID Global
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= totalThreads) return;

    // 2. ID Geometría (Cíclico si batch > 1)
    int geo_id = id % cellCount;

    // 3. Carga de Datos (Coalesced)
    float H = d_initialDepths[id];
    const float Q_target = d_targetDischarges[id];

    // Invariantes Geométricos
    const float b = d_bottomWidths[geo_id];
    const float m = d_sideSlopes[geo_id];
    const float n = d_manningCoeffs[geo_id];
    const float S = d_bedSlopes[geo_id];

    // 4. Pre-cálculos Matemáticos (Fuera del bucle)
    const float sqrt_slope = sqrtf(S);
    const float pythagoras = sqrtf(1.0f + m * m);
    const float inv_n = 1.0f / n;

    // 5. Newton-Raphson (Unrolled)
    #pragma unroll
    for (int i = 0; i < MAX_ITERATIONS; i++) {
        // Geometría actual
        float A = device_calculateA(H, b, m);
        float P = device_calculateP_optimized(H, b, pythagoras);
        float T = device_calculateTopWidth(H, b, m);

        // Función y Derivada
        float Q_calc = device_calculateQ_opt(A, P, inv_n, sqrt_slope);
        float f = Q_calc - Q_target;
        float df = device_calculate_dQ_dH_opt(A, P, T, pythagoras, Q_calc);

        // Paso seguro (Branchless safe division)
        // Si df es 0, usamos epsilon con el signo correcto para empujar H
        float df_safe = df + copysignf(SAFE_EPSILON, df);

        float H_next = H - f / df_safe;

        // Clamp (Branchless)
        H = fmaxf(MIN_DEPTH, H_next);
    }

    // 6. Resultado Final
    float A_final = device_calculateA(H, b, m);
    // Velocidad = Q / A.
    // Usamos fmaxf en A por seguridad extrema, aunque H >= MIN_DEPTH implica A > 0.
    float V = Q_target / fmaxf(A_final, SAFE_EPSILON);

    // Escritura Intercalada [H, V]
    int out_idx = id * 2;
    d_results[out_idx]     = H;
    d_results[out_idx + 1] = V;
}

// --- Launcher ---
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

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

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