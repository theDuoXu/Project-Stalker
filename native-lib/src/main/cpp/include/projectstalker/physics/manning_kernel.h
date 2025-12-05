// src/main/cpp/include/projectstalker/physics/manning_kernel.h
#pragma once

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
    int cellCount
);

// --- Smart Solver ---
// Calcula todo el dominio, pero asume lógica "Smart Fetch" para inputs.
// Ideal para sistemas de alerta temprana y entradas de ondas nuevas.
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

// --- Full Evolution Solver ---
// Estructura separada para simulación completa.
// Aunque por ahora comparte inputs, este kernel está preparado para:
// 1. Lógica de caudal compleja (interacción de olas).
// 2. Outputs masivos (sin recorte triangular).
void launchManningFullKernel(
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

#ifdef __cplusplus
}
#endif