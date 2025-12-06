// src/main/cpp/include/projectstalker/physics/manning_kernel.h
#pragma once

#include <cuda_runtime_api.h> // Necesario para el tipo cudaStream_t

#ifdef __cplusplus
extern "C" {
#endif

// --- Baking (Pre-cálculo) ---
void launchManningBakingKernel(
    float* d_inv_n,
    float* d_sqrt_slope,
    float* d_pythagoras,
    const float* d_manning,
    const float* d_bedSlope,
    const float* d_sideSlope,
    int cellCount,
    cudaStream_t stream // <--- Añadido para init asíncrono
);

// --- Smart Solver (Lazy / Optimized) ---
// Se mantiene SIN stream para ejecución directa (Legacy/Default stream).
void launchManningSmartKernel(
    float* d_results,
    const float* d_newInflows,
    const float* d_initialQ,
    const float* d_initialDepths,
    const float* d_bottomWidths,
    const float* d_sideSlopes,
    const float* d_inv_n,
    const float* d_sqrt_slope,
    const float* d_pythagoras,
    int batchSize,
    int cellCount
);

// --- Step Solver (Full Evolution / Ping-Pong) ---
// Calcula UN SOLO paso de tiempo (t) basándose en el estado anterior (t-1).
// Se llama repetidamente desde el bucle C++.
//
// @param d_current_Q     Buffer de escritura para el estado Q actual (Ping/Pong).
// @param d_prev_Q        Buffer de lectura del estado Q anterior (Ping/Pong).
// @param d_results       Buffer global de salida (se escribe solo si save_to_output es true).
// @param current_inflow  Caudal de entrada para este paso de tiempo (t).
// @param write_offset    Offset en d_results donde escribir (si corresponde).
// @param save_to_output  Flag para indicar si este paso debe guardarse (Stride).
// @param stream          Stream de CUDA donde encolar la ejecución (Necesario para Graphs).
void launchManningStepKernel(
    float* d_current_Q,
    const float* d_prev_Q,
    float* d_results,
    float current_inflow,
    const float* d_initialDepths, // Semilla
    const float* d_bottomWidths,
    const float* d_sideSlopes,
    const float* d_inv_n,
    const float* d_sqrt_slope,
    const float* d_pythagoras,
    int cellCount,
    bool save_to_output,
    int write_idx_h,
    int write_idx_v,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif