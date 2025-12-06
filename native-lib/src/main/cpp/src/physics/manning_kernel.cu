#include "projectstalker/physics/manning_kernel.h"
#include <cuda_runtime.h>
#include <cmath>

// --- Constantes de Simulación ---
#define MAX_ITERATIONS 5
#define SAFE_EPSILON 1e-7f
#define MIN_DEPTH 0.001f

// --- Constantes Matemáticas Pre-calculadas ---
#define ONE_THIRD 0.33333333f
#define FIVE_THIRDS 1.66666667f
#define FOUR_THIRDS 1.33333333f
#define TWO_THIRDS 0.66666667f

// =============================================================================
// HELPERS MATEMÁTICOS (Inline Device Functions)
// =============================================================================

__device__ inline float device_calculateA(float H, float b, float m) {
    return (b + m * H) * H;
}

__device__ inline float device_calculateP_optimized(float H, float b, float pythagoras) {
    return b + 2.0f * H * pythagoras;
}

__device__ inline float device_calculateTopWidth(float H, float b, float m) {
    return b + 2.0f * m * H;
}

__device__ inline float device_calculateQ_opt(float A, float P, float inv_n, float sqrt_slope) {
    float R = A / fmaxf(P, SAFE_EPSILON);
    return inv_n * A * powf(R, TWO_THIRDS) * sqrt_slope;
}

__device__ inline float device_calculate_dQ_dH_opt(float A, float P, float T, float pythagoras, float Q) {
    float term_A = (FIVE_THIRDS * T) / fmaxf(A, SAFE_EPSILON);
    float term_P = (FOUR_THIRDS * pythagoras) / fmaxf(P, SAFE_EPSILON);
    return Q * (term_A - term_P);
}

// =============================================================================
// CORE PHYSICS
// =============================================================================
__device__ inline float device_solve_manning_cell(
    float Q_target,
    float H_initial_seed,
    float b,
    float m,
    float inv_n,
    float sqrt_slope,
    float pythagoras
) {
    float H = H_initial_seed;

    #pragma unroll
    for (int i = 0; i < MAX_ITERATIONS; i++) {
        float A = device_calculateA(H, b, m);
        float P = device_calculateP_optimized(H, b, pythagoras);
        float T = device_calculateTopWidth(H, b, m);

        float Q_calc = device_calculateQ_opt(A, P, inv_n, sqrt_slope);
        float f = Q_calc - Q_target;
        float df = device_calculate_dQ_dH_opt(A, P, T, pythagoras, Q_calc);

        float df_safe = df + copysignf(SAFE_EPSILON, df);
        float H_next = H - f / df_safe;

        H = fmaxf(MIN_DEPTH, H_next);
    }
    return H;
}

// =============================================================================
// KERNEL 1: BAKING
// =============================================================================
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

// =============================================================================
// KERNEL 2: SMART SOLVER
// =============================================================================
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

    const float b = d_bottomWidths[cell_idx];
    const float m = d_sideSlopes[cell_idx];
    const float inv_n = d_inv_n[cell_idx];
    const float sqrt_slope = d_sqrt_slope[cell_idx];
    const float pythagoras = d_pythagoras[cell_idx];

    float Q_target;
    if (cell_idx <= step_idx) {
        Q_target = d_newInflows[step_idx - cell_idx];
    } else {
        Q_target = d_initialQ[cell_idx - (step_idx + 1)];
    }

    float H = d_initialDepths[cell_idx];
    H = device_solve_manning_cell(Q_target, H, b, m, inv_n, sqrt_slope, pythagoras);

    float A_final = device_calculateA(H, b, m);
    float V_final = Q_target / fmaxf(A_final, SAFE_EPSILON);

    d_results[id]                = H;
    d_results[id + totalThreads] = V_final;
}

// =============================================================================
// KERNEL 3: STEP SOLVER (Refactorizado para Graphs)
// =============================================================================
__global__ void manningStepKernel(
    float* __restrict__ d_current_Q,
    const float* __restrict__ d_prev_Q,
    float* __restrict__ d_results,
    float current_inflow,
    const float* __restrict__ d_initialDepths,
    const float* __restrict__ d_bottomWidths,
    const float* __restrict__ d_sideSlopes,
    const float* __restrict__ d_inv_n,
    const float* __restrict__ d_sqrt_slope,
    const float* __restrict__ d_pythagoras,
    int cellCount,
    bool save_to_output,
    int write_idx_h,
    int write_idx_v
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= cellCount) return;

    // 1. Advección
    float my_Q;
    if (id == 0) {
        my_Q = current_inflow;
    } else {
        my_Q = d_prev_Q[id - 1];
    }
    d_current_Q[id] = my_Q; // Ping-Pong save

    // 2. Geometría
    const float b = d_bottomWidths[id];
    const float m = d_sideSlopes[id];
    const float inv_n = d_inv_n[id];
    const float sqrt_slope = d_sqrt_slope[id];
    const float pythagoras = d_pythagoras[id];

    // 3. Resolución
    float H = device_solve_manning_cell(my_Q, d_initialDepths[id], b, m, inv_n, sqrt_slope, pythagoras);

    // 4. Output Condicional
    if (save_to_output) {
        float A_final = device_calculateA(H, b, m);
        float V_final = my_Q / fmaxf(A_final, SAFE_EPSILON);

        d_results[write_idx_h + id] = H;
        d_results[write_idx_v + id] = V_final;
    }
}

// =============================================================================
// LAUNCHERS (Interfaz C++)
// =============================================================================

// Baking usa Stream para permitir inicialización asíncrona
void launchManningBakingKernel(float* d_inv_n, float* d_sqrt_slope, float* d_pythagoras, const float* d_manning, const float* d_bedSlope, const float* d_sideSlope, int cellCount, cudaStream_t stream) {
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (cellCount + threadsPerBlock - 1) / threadsPerBlock;
    manningBakingKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_inv_n, d_sqrt_slope, d_pythagoras, d_manning, d_bedSlope, d_sideSlope, cellCount);
}

void launchManningSmartKernel(float* d_results, const float* d_newInflows, const float* d_initialQ, const float* d_initialDepths, const float* d_bottomWidths, const float* d_sideSlopes, const float* d_inv_n, const float* d_sqrt_slope, const float* d_pythagoras, int batchSize, int cellCount) {
    int totalThreads = batchSize * cellCount;
    if (totalThreads == 0) return;
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;
    // Lanzamiento tradicional (Default Stream)
    manningSmartKernel<<<blocksPerGrid, threadsPerBlock>>>(d_results, d_newInflows, d_initialQ, d_initialDepths, d_bottomWidths, d_sideSlopes, d_inv_n, d_sqrt_slope, d_pythagoras, totalThreads, cellCount);
}

// STEP: Acepta Stream para permitir Captura de Grafos
void launchManningStepKernel(
    float* d_current_Q, const float* d_prev_Q, float* d_results,
    float current_inflow, const float* d_initialDepths,
    const float* d_bottomWidths, const float* d_sideSlopes,
    const float* d_inv_n, const float* d_sqrt_slope, const float* d_pythagoras,
    int cellCount, bool save_to_output, int write_idx_h, int write_idx_v,
    cudaStream_t stream
) {
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (cellCount + threadsPerBlock - 1) / threadsPerBlock;

    manningStepKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_current_Q, d_prev_Q, d_results,
        current_inflow, d_initialDepths,
        d_bottomWidths, d_sideSlopes, d_inv_n, d_sqrt_slope, d_pythagoras,
        cellCount, save_to_output, write_idx_h, write_idx_v
    );
}