// manning_kernel.cu
#include "projectstalker/physics/manning_kernel.h"
#include <cuda_runtime.h>
#include <cmath> // para fmaxf, sqrtf, powf, copysignf

// --- Constantes del Kernel ---
#define MAX_ITERATIONS 20
#define SAFE_EPSILON 1e-7f
#define MIN_DEPTH 0.001f

// --- 1. Helpers Geométricos (Restaurados para legibilidad) ---

__device__ inline float device_calculateA(float H, float b, float m) {
    return (b + m * H) * H;
}

__device__ inline float device_calculateP_optimized(float H, float b, float pythagoras) {
    return b + 2.0f * H * pythagoras;
}

__device__ inline float device_calculateTopWidth(float H, float b, float m) {
    return b + 2.0f * m * H;
}

// --- 2. Helpers Hidráulicos (Optimizados) ---

/**
 * Calcula Q recibiendo A y P ya calculados (evita recálculo).
 */
__device__ inline float device_calculateQ_from_state(
    float A,
    float P,
    float inv_n,
    float sqrt_slope
) {
    // R = A / P
    // Q = (1/n) * A * R^(2/3) * sqrt(S)
    // Optimización: powf es costoso, pero R^(2/3) es inevitable.
    float R = A / P;
    return inv_n * A * powf(R, 0.6666667f) * sqrt_slope;
}

/**
 * Calcula dQ/dH recibiendo el estado geométrico completo.
 */
__device__ inline float device_calculate_dQ_dH_from_state(
    float A,
    float P,
    float topWidth,
    float pythagoras,
    float currentQ
) {
    // Término A: (5/3) * (T / A)
    float term_A = (1.6666667f * topWidth) / A;

    // Término P: (4/3) * (FactorPitágoras / P)
    // Nota: La derivada del perímetro dP/dH es 2 * pythagoras.
    // La fórmula es -(2/3)*(1/P)*(dP/dH).
    // (2/3) * 2 = 4/3.
    float term_P = (1.3333333f * pythagoras) / P;

    return currentQ * (term_A - term_P);
}

// --- 3. Kernel Principal ---

__global__ void manningSolverKernel(
    float* __restrict__ d_results,
    const float* __restrict__ d_initialDepths,
    const float* __restrict__ d_targetDischarges,
    const float* __restrict__ d_bottomWidths,
    const float* __restrict__ d_sideSlopes,
    const float* __restrict__ d_manningCoeffs,
    const float* __restrict__ d_bedSlopes,
    int totalThreads,
    int cellCount
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= totalThreads) return;

    int geo_id = id % cellCount;

    // Carga de datos (Coalesced access)
    float H = d_initialDepths[id];
    const float Q_target = d_targetDischarges[id];
    const float b = d_bottomWidths[geo_id];
    const float m = d_sideSlopes[geo_id];
    const float n = d_manningCoeffs[geo_id];
    const float S = d_bedSlopes[geo_id];

    // --- PRE-CÁLCULOS INVARIANTES (Fuera del bucle) ---
    const float sqrt_slope = sqrtf(S);
    const float pythagoras = sqrtf(1.0f + m * m);
    const float inv_n = 1.0f / n;

    // --- BUCLE DE NEWTON ---
    #pragma unroll
    for (int i = 0; i < MAX_ITERATIONS; i++) {

        // 1. Calcular Geometría (Una sola vez por iteración usando helpers)
        float A = device_calculateA(H, b, m);
        float P = device_calculateP_optimized(H, b, pythagoras);
        float T = device_calculateTopWidth(H, b, m);

        // 2. Calcular Hidráulica (Pasando los valores ya calculados)
        float Q_calc = device_calculateQ_from_state(A, P, inv_n, sqrt_slope);
        float f_H = Q_calc - Q_target;

        float dQ_dH = device_calculate_dQ_dH_from_state(A, P, T, pythagoras, Q_calc);

        // 3. Paso de Newton (Branchless safe division)
        float dQ_dH_safe = dQ_dH + copysignf(SAFE_EPSILON, dQ_dH);
        float H_next = H - f_H / dQ_dH_safe;

        // 4. Saneamiento (Branchless)
        H = fmaxf(MIN_DEPTH, H_next);
    }

    // --- CÁLCULO FINAL ---
    // Reutilizamos el helper para el resultado final de V
    float A_final = device_calculateA(H, b, m);
    float V = Q_target / A_final;

    int result_idx = id * 2;
    d_results[result_idx] = H;
    d_results[result_idx + 1] = V;
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