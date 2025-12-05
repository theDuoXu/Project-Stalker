// src/main/cpp/src/physics/manning_kernel.cu
#include "projectstalker/physics/manning_kernel.h"
#include <cuda_runtime.h>
#include <cmath>

// --- Constantes de Simulación ---
#define MAX_ITERATIONS 20
#define SAFE_EPSILON 1e-7f
#define MIN_DEPTH 0.001f

// --- Constantes Matemáticas Pre-calculadas ---
// Optimizaciones para evitar divisiones costosas en el bucle principal
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

// Usa la constante de Pitágoras pre-calculada: sqrt(1 + m^2)
__device__ inline float device_calculateP_optimized(float H, float b, float pythagoras) {
    return b + 2.0f * H * pythagoras;
}

__device__ inline float device_calculateTopWidth(float H, float b, float m) {
    return b + 2.0f * m * H;
}

// Calcula el Caudal (Q) usando Manning. R^(2/3) es la operación pesada (powf).
__device__ inline float device_calculateQ_opt(float A, float P, float inv_n, float sqrt_slope) {
    float R = A / fmaxf(P, SAFE_EPSILON);
    return inv_n * A * powf(R, TWO_THIRDS) * sqrt_slope;
}

// Calcula la derivada dQ/dH factorizada para Newton-Raphson
__device__ inline float device_calculate_dQ_dH_opt(float A, float P, float T, float pythagoras, float Q) {
    float term_A = (FIVE_THIRDS * T) / fmaxf(A, SAFE_EPSILON);
    float term_P = (FOUR_THIRDS * pythagoras) / fmaxf(P, SAFE_EPSILON);
    return Q * (term_A - term_P);
}

// =============================================================================
// CORE PHYSICS: SOLVER NUMÉRICO
// =============================================================================
// Resuelve la ecuación f(H) = Q_calc(H) - Q_target = 0 para encontrar H.
// Esta función es agnóstica a la estrategia de simulación.
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

        // Estabilización numérica: evita división por cero en df
        float df_safe = df + copysignf(SAFE_EPSILON, df);
        float H_next = H - f / df_safe;

        H = fmaxf(MIN_DEPTH, H_next);
    }
    return H;
}

// =============================================================================
// KERNEL 1: BAKING (Pre-cálculo de Geometría)
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
// KERNEL 2: SMART SOLVER (Estrategia Optimizada)
// =============================================================================
// Calcula todo el batch de golpe asumiendo propagación de onda sobre estado estable.
// Ideal para alertas tempranas y respuesta rápida.
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

    // 1. Carga de Geometría
    const float b = d_bottomWidths[cell_idx];
    const float m = d_sideSlopes[cell_idx];
    const float inv_n = d_inv_n[cell_idx];
    const float sqrt_slope = d_sqrt_slope[cell_idx];
    const float pythagoras = d_pythagoras[cell_idx];

    // 2. Selección de Caudal (Smart Fetch)
    float Q_target;
    if (cell_idx <= step_idx) {
        Q_target = d_newInflows[step_idx - cell_idx];
    } else {
        Q_target = d_initialQ[cell_idx - (step_idx + 1)];
    }

    // 3. Resolución Física
    float H = d_initialDepths[cell_idx]; // Seed constante
    H = device_solve_manning_cell(Q_target, H, b, m, inv_n, sqrt_slope, pythagoras);

    // 4. Escritura de Resultados (SoA)
    float A_final = device_calculateA(H, b, m);
    float V_final = Q_target / fmaxf(A_final, SAFE_EPSILON);

    d_results[id]                = H;
    d_results[id + totalThreads] = V_final;
}

// =============================================================================
// KERNEL 3: STEP SOLVER (Estrategia Full Evolution / Ping-Pong)
// =============================================================================
// Calcula UN solo paso de tiempo para todo el dominio.
// Lee del estado anterior (d_prev_Q) y escribe en el actual (d_current_Q).
// Soporta escritura condicional (Stride) para ahorrar ancho de banda PCIe.
__global__ void manningStepKernel(
    float* __restrict__ d_current_Q,        // Escritura (Estado t)
    const float* __restrict__ d_prev_Q,     // Lectura (Estado t-1)
    float* __restrict__ d_results,          // Output Global (Opcional)
    float current_inflow,                   // Caudal de entrada en celda 0
    const float* __restrict__ d_initialDepths,
    const float* __restrict__ d_bottomWidths,
    const float* __restrict__ d_sideSlopes,
    const float* __restrict__ d_inv_n,
    const float* __restrict__ d_sqrt_slope,
    const float* __restrict__ d_pythagoras,
    int cellCount,
    bool save_to_output,                    // Flag para Stride
    int write_idx_h,                        // Índice base escritura H
    int write_idx_v                         // Índice base escritura V
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= cellCount) return;

    // 1. ADVECCIÓN / PROPAGACIÓN (Onda Cinemática)
    // El agua se mueve de celda i-1 a celda i.
    // Causa la divergencia de UN ÚNICO y SINGULAR WARP (el que contenga la celda inicial), aceptable
    float my_Q;
    if (id == 0) {
        my_Q = current_inflow; // Condición de frontera (entrada al río)
    } else {
        my_Q = d_prev_Q[id - 1]; // Propagación desde arriba
    }

    // Guardamos el estado para el siguiente paso (Ping-Pong)
    // Esto es crucial: aunque no guardemos el output para Java,
    // necesitamos persistir el estado Q en VRAM para el paso t+1.
    d_current_Q[id] = my_Q;

    // 2. CARGA DE GEOMETRÍA
    const float b = d_bottomWidths[id];
    const float m = d_sideSlopes[id];
    const float inv_n = d_inv_n[id];
    const float sqrt_slope = d_sqrt_slope[id];
    const float pythagoras = d_pythagoras[id];

    // 3. RESOLUCIÓN FÍSICA (Manning Dinámico)
    // Usamos la profundidad base como semilla para estabilidad
    float H = device_solve_manning_cell(my_Q, d_initialDepths[id], b, m, inv_n, sqrt_slope, pythagoras);

    // 4. ESCRITURA CONDICIONAL (Output Stride)
    // Solo escribimos en el buffer de salida masivo si la CPU lo solicita.
    // NO causa divergencia de warps porque todos los warps se van por TRUE o por FALSE

    if (save_to_output) {
        float A_final = device_calculateA(H, b, m);
        float V_final = my_Q / fmaxf(A_final, SAFE_EPSILON);

        // Escritura SoA usando los índices pre-calculados por la CPU
        // emite una instrucción ST.E y NO se bloquea esperando a que el dato viaje hasta la VRAM
        // fire and forget para el hilo
        d_results[write_idx_h + id] = H;
        d_results[write_idx_v + id] = V_final;
    }
}

// =============================================================================
// LAUNCHERS (Interfaz C++)
// =============================================================================

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

// Launcher para la estrategia Paso a Paso
void launchManningStepKernel(
    float* d_current_Q, const float* d_prev_Q, float* d_results,
    float current_inflow, const float* d_initialDepths,
    const float* d_bottomWidths, const float* d_sideSlopes,
    const float* d_inv_n, const float* d_sqrt_slope, const float* d_pythagoras,
    int cellCount, bool save_to_output, int write_idx_h, int write_idx_v
) {
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (cellCount + threadsPerBlock - 1) / threadsPerBlock;

    manningStepKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_current_Q, d_prev_Q, d_results,
        current_inflow, d_initialDepths,
        d_bottomWidths, d_sideSlopes, d_inv_n, d_sqrt_slope, d_pythagoras,
        cellCount, save_to_output, write_idx_h, write_idx_v
    );
}