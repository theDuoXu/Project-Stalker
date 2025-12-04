// src/main/cpp/include/projectstalker/physics/manning_solver.h
#pragma once

#include <vector>
#include <cstddef>
#include <cuda_runtime.h>

/**
 * Estructura de Sesión para Manning (Stateful).
 * OPTIMIZADA PARA RTX 5090:
 * Incluye gestión de Streams Asíncronos para maximizar el throughput del bus PCIe.
 */
struct ManningSession {
    // --- 0. Control de Ejecución (Async) ---
    cudaStream_t stream = nullptr; // Stream dedicado de alta prioridad

    // --- 1. Geometría Invariante ---
    float* d_bottomWidths = nullptr;
    float* d_sideSlopes   = nullptr;
    float* d_inv_n        = nullptr;
    float* d_sqrt_slope   = nullptr;
    float* d_pythagoras   = nullptr;

    // --- 2. Estado Base (Flyweight Intrinsic) ---
    float* d_initialQ      = nullptr;
    float* d_initialDepths = nullptr;

    // --- 3. Buffers de Trabajo Adaptativos ---
    float* d_results       = nullptr;
    float* d_newInflows    = nullptr;

    // --- 4. Metadatos ---
    size_t resultCapacityElements = 0;
    size_t inputBatchCapacity     = 0;
    int cellCount = 0;
};

// ... (Las firmas de funciones init/run/destroy se mantienen iguales)
ManningSession* init_manning_session(
    const float* h_bottomWidths,
    const float* h_sideSlopes,
    const float* h_manningCoeffs,
    const float* h_bedSlopes,
    const float* h_initialDepths,
    const float* h_initialQ,
    int cellCount
);

void run_manning_batch_stateful(
    ManningSession* session,
    const float* h_pinned_inflows,
    float* h_pinned_results,
    int batchSize
);

void destroy_manning_session(ManningSession* session);