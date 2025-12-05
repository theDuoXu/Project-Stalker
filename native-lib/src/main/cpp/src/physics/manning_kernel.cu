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

// --- Helpers Matemáticos (Inalterados) ---

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
    // Evitamos división por cero con fmaxf en P
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
    float term_A = (FIVE_THIRDS * T) / fmaxf(A, SAFE_EPSILON);
    float term_P = (FOUR_THIRDS * pythagoras) / fmaxf(P, SAFE_EPSILON);

    return Q * (term_A - term_P);
}

// -----------------------------------------------------------------------------
// CORE PHYSICS: DEVICE SOLVER
// -----------------------------------------------------------------------------
// Extraemos la lógica de Newton-Raphson para reutilizarla en ambos kernels.
// Esto garantiza que la física sea idéntica en modo Smart y Full.
__device__ inline float device_solve_manning_cell(
    float Q_target,
    float H_initial,
    float b,
    float m,
    float inv_n,
    float sqrt_slope,
    float pythagoras
) {
    float H = H_initial;

    #pragma unroll
    for (int i = 0; i < MAX_ITERATIONS; i++) {
        float A = device_calculateA(H, b, m);
        float P = device_calculateP_optimized(H, b, pythagoras);
        float T = device_calculateTopWidth(H, b, m);

        float Q_calc = device_calculateQ_opt(A, P, inv_n, sqrt_slope);
        float f = Q_calc - Q_target;
        float df = device_calculate_dQ_dH_opt(A, P, T, pythagoras, Q_calc);

        // Estabilización numérica
        float df_safe = df + copysignf(SAFE_EPSILON, df);
        float H_next = H - f / df_safe;

        H = fmaxf(MIN_DEPTH, H_next);
    }
    return H;
}

// -----------------------------------------------------------------------------
// KERNEL 1: BAKING (Pre-cálculo de Geometría)
// -----------------------------------------------------------------------------
__global__ void manningBakingKernel(
    float* __restrict__ d_inv_n,
    float* __restrict__ d_sqrt_slope,
    float* __restrict__ d_pythagoras,
    const float* __restrict__ d_manning,
    const float* __restrict__ d_bedSlope,
    const float* __restrict__ d_sideSlope,
    int cellCount
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= cellCount) return;

    d_inv_n[id] = 1.0f / d_manning[id];
    d_sqrt_slope[id] = sqrtf(d_bedSlope[id]);
    float m = d_sideSlope[id];
    d_pythagoras[id] = sqrtf(1.0f + m * m);
}

// -----------------------------------------------------------------------------
// KERNEL 2: SMART SOLVER (Stateful + Smart Fetch + SoA Writing)
// -----------------------------------------------------------------------------
// Versión optimizada para arranque de simulación (Alerta Temprana).
// Asume que la zona aguas abajo es estable (Steady State).
__global__ void manningSmartKernel(
    float* __restrict__ d_results,
    const float* __restrict__ d_newInflows,
    const float* __restrict__ d_initialQ,
    const float* __restrict__ d_initialDepths,
    const float* __restrict__ d_bottomWidths,
    const float* __restrict__ d_sideSlopes,
    const float* __restrict__ d_inv_n,
    const float* __restrict__ d_sqrt_slope,
    const float* __restrict__ d_pythagoras,
    int totalThreads,
    int cellCount
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= totalThreads) return;

    int cell_idx = id % cellCount;
    int step_idx = id / cellCount;

    // 1. Carga Geometría
    const float b = d_bottomWidths[cell_idx];
    const float m = d_sideSlopes[cell_idx];
    const float inv_n = d_inv_n[cell_idx];
    const float sqrt_slope = d_sqrt_slope[cell_idx];
    const float pythagoras = d_pythagoras[cell_idx];

    // 2. Smart Fetch
    float Q_target;
    if (cell_idx <= step_idx) {
        Q_target = d_newInflows[step_idx - cell_idx];
    } else {
        Q_target = d_initialQ[cell_idx - (step_idx + 1)];
    }

    // 3. Física (Reutilizada)
    float H = d_initialDepths[cell_idx]; // Seed
    H = device_solve_manning_cell(Q_target, H, b, m, inv_n, sqrt_slope, pythagoras);

    // 4. Escritura SoA
    float A_final = device_calculateA(H, b, m);
    float V = Q_target / fmaxf(A_final, SAFE_EPSILON);

    d_results[id]                = H;
    d_results[id + totalThreads] = V;
}

// -----------------------------------------------------------------------------
// KERNEL 3: FULL EVOLUTION SOLVER
// -----------------------------------------------------------------------------
// Versión robusta para simulación científica completa.
// Preparada para lógica de interacción de olas (aunque por ahora usa inputs simples).
// Se diferencia en que está aislada para permitir transferencias masivas sin tocar la lógica Smart.
__global__ void manningFullKernel(
    float* __restrict__ d_results,
    const float* __restrict__ d_newInflows,
    const float* __restrict__ d_initialQ,
    const float* __restrict__ d_initialDepths,
    const float* __restrict__ d_bottomWidths,
    const float* __restrict__ d_sideSlopes,
    const float* __restrict__ d_inv_n,
    const float* __restrict__ d_sqrt_slope,
    const float* __restrict__ d_pythagoras,
    int totalThreads,
    int cellCount
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= totalThreads) return;

    int cell_idx = id % cellCount;
    int step_idx = id / cellCount;

    // 1. Geometría
    const float b = d_bottomWidths[cell_idx];
    const float m = d_sideSlopes[cell_idx];
    const float inv_n = d_inv_n[cell_idx];
    const float sqrt_slope = d_sqrt_slope[cell_idx];
    const float pythagoras = d_pythagoras[cell_idx];

    // 2. Full Evolution Logic (Placeholder compatible)
    // En el futuro, aquí leeremos una matriz Q completa o calcularemos Q(i, t) = f(Q(i-1, t)).
    float Q_target;
    if (cell_idx <= step_idx) {
        Q_target = d_newInflows[step_idx - cell_idx];
    } else {
        Q_target = d_initialQ[cell_idx - (step_idx + 1)];
    }

    // 3. Física (Reutilizada)
    float H = d_initialDepths[cell_idx];
    H = device_solve_manning_cell(Q_target, H, b, m, inv_n, sqrt_slope, pythagoras);

    // 4. Escritura SoA
    float A_final = device_calculateA(H, b, m);
    float V = Q_target / fmaxf(A_final, SAFE_EPSILON);

    d_results[id]                = H;
    d_results[id + totalThreads] = V;
}

// --- Launchers (Implementación de manning_kernel.h) ---

void launchManningBakingKernel(float* d_inv_n, float* d_sqrt_slope, float* d_pythagoras, const float* d_manning, const float* d_bedSlope, const float* d_sideSlope, int cellCount) {
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (cellCount + threadsPerBlock - 1) / threadsPerBlock;
    manningBakingKernel<<<blocksPerGrid, threadsPerBlock>>>(d_inv_n, d_sqrt_slope, d_pythagoras, d_manning, d_bedSlope, d_sideSlope, cellCount);
}

void launchManningSmartKernel(float* d_results, const float* d_newInflows, const float* d_initialQ, const float* d_initialDepths, const float* d_bottomWidths, const float* d_sideSlopes, const float* d_inv_n, const float* d_sqrt_slope, const float* d_pythagoras, int batchSize, int cellCount) {
    int totalThreads = batchSize * cellCount;
    if (totalThreads == 0) return;
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;
    manningSmartKernel<<<blocksPerGrid, threadsPerBlock>>>(d_results, d_newInflows, d_initialQ, d_initialDepths, d_bottomWidths, d_sideSlopes, d_inv_n, d_sqrt_slope, d_pythagoras, totalThreads, cellCount);
}

void launchManningFullKernel(float* d_results, const float* d_newInflows, const float* d_initialQ, const float* d_initialDepths, const float* d_bottomWidths, const float* d_sideSlopes, const float* d_inv_n, const float* d_sqrt_slope, const float* d_pythagoras, int batchSize, int cellCount) {
    int totalThreads = batchSize * cellCount;
    if (totalThreads == 0) return;
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;
    manningFullKernel<<<blocksPerGrid, threadsPerBlock>>>(d_results, d_newInflows, d_initialQ, d_initialDepths, d_bottomWidths, d_sideSlopes, d_inv_n, d_sqrt_slope, d_pythagoras, totalThreads, cellCount);
}