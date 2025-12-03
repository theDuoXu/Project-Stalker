// src/main/cpp/src/physics/manning_solver.cpp
#include "projectstalker/physics/manning_solver.h"
#include "projectstalker/physics/manning_kernel.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sstream>

// Macro de Chequeo de Errores CUDA
#define CUDA_CHECK_M(call) { \
    cudaError_t err_code = call; \
    if (err_code != cudaSuccess) { \
        std::stringstream ss; \
        ss << "CUDA Error en Manning Solver " << __FILE__ << ":" << __LINE__ \
           << " | Código: " << err_code \
           << " | Mensaje: " << cudaGetErrorString(err_code); \
        std::cerr << ss.str() << std::endl; \
        throw std::runtime_error(ss.str()); \
    } \
}

// -----------------------------------------------------------------------------
// IMPLEMENTACIÓN DEL CICLO DE VIDA (Stateful)
// -----------------------------------------------------------------------------

ManningSession* init_manning_session(
    const float* h_bottomWidths,
    const float* h_sideSlopes,
    const float* h_manningCoeffs,
    const float* h_bedSlopes,
    int cellCount
) {
    if (cellCount <= 0) return nullptr;

    // 1. Crear la estructura de sesión
    ManningSession* session = new ManningSession();
    session->cellCount = cellCount;
    size_t cellBytes = cellCount * sizeof(float);

    // Buffers temporales para Baking (se liberan antes de salir)
    float* d_temp_manning = nullptr;
    float* d_temp_bed = nullptr;

    try {
        // 2. Reserva VRAM para Geometría Invariante
        CUDA_CHECK_M(cudaMalloc(&session->d_bottomWidths, cellBytes));
        CUDA_CHECK_M(cudaMalloc(&session->d_sideSlopes, cellBytes));

        CUDA_CHECK_M(cudaMalloc(&session->d_inv_n, cellBytes));
        CUDA_CHECK_M(cudaMalloc(&session->d_sqrt_slope, cellBytes));
        CUDA_CHECK_M(cudaMalloc(&session->d_pythagoras, cellBytes));

        // 3. Reserva VRAM para Buffers de Estado del Río (Tamaño Fijo)
        // Validado: Se crean en init para evitar mallocs en el bucle principal
        CUDA_CHECK_M(cudaMalloc(&session->d_initialQ, cellBytes));
        CUDA_CHECK_M(cudaMalloc(&session->d_initialDepths, cellBytes));

        // 4. Reserva Temporales para Baking
        CUDA_CHECK_M(cudaMalloc(&d_temp_manning, cellBytes));
        CUDA_CHECK_M(cudaMalloc(&d_temp_bed, cellBytes));

        // 5. Copia Host -> Device (Geometría Cruda)
        CUDA_CHECK_M(cudaMemcpy(session->d_bottomWidths, h_bottomWidths, cellBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK_M(cudaMemcpy(session->d_sideSlopes, h_sideSlopes, cellBytes, cudaMemcpyHostToDevice));
        // Copiamos a temporales los que solo sirven para baking
        CUDA_CHECK_M(cudaMemcpy(d_temp_manning, h_manningCoeffs, cellBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK_M(cudaMemcpy(d_temp_bed, h_bedSlopes, cellBytes, cudaMemcpyHostToDevice));

        // 6. BAKING: Pre-calcular constantes físicas en GPU
        launchManningBakingKernel(
            session->d_inv_n,
            session->d_sqrt_slope,
            session->d_pythagoras,
            d_temp_manning,
            d_temp_bed,
            session->d_sideSlopes,
            cellCount
        );
        CUDA_CHECK_M(cudaDeviceSynchronize());

        // 7. Limpieza de temporales
        cudaFree(d_temp_manning);
        cudaFree(d_temp_bed);

        return session;

    } catch (...) {
        // Rollback en caso de error en inicialización
        if (d_temp_manning) cudaFree(d_temp_manning);
        if (d_temp_bed) cudaFree(d_temp_bed);
        destroy_manning_session(session);
        throw; // Relanzar excepción
    }
}

// Helper interno para redimensionar con histéresis
static void resize_buffer_if_needed(float** ptr, size_t* currentCap, size_t needed) {
    // Lógica Balloning:
    // 1. Si no cabe (needed > current) -> Crecer
    // 2. Si sobra demasiado (current > needed * 2) -> Encoger (Shrink to fit)
    if (needed > *currentCap || *currentCap > (needed * 2)) {
        if (*ptr) {
            cudaFree(*ptr);
            *ptr = nullptr;
        }

        // Asignamos lo necesario + 20% de buffer para evitar reallocs frecuentes por pequeñas variaciones
        size_t newCap = (size_t)(needed * 1.2f);

        // Caso borde: Si needed es 0, newCap es 0
        if (newCap > 0) {
            cudaError_t err = cudaMalloc(ptr, newCap * sizeof(float));
            if (err != cudaSuccess) {
                // Si falla el +20%, intentamos con lo justo (fallback)
                newCap = needed;
                CUDA_CHECK_M(cudaMalloc(ptr, newCap * sizeof(float)));
            }
        }
        *currentCap = newCap;
    }
}

std::vector<float> run_manning_batch_stateful(
    ManningSession* session,
    const float* h_newInflows,
    const float* h_initialDepths,
    const float* h_initialQ,
    int batchSize
) {
    if (!session || batchSize <= 0) return {};

    int cellCount = session->cellCount;
    size_t cellBytes = cellCount * sizeof(float);

    // --- 1. Gestión de Memoria Adaptativa (Inputs & Outputs) ---
    size_t neededInputElements = (size_t)batchSize;
    // El output es gigante: Batch * Celdas * 2 (H y V)
    size_t neededOutputElements = (size_t)batchSize * cellCount * 2;

    resize_buffer_if_needed(&session->d_newInflows, &session->inputBatchCapacity, neededInputElements);
    resize_buffer_if_needed(&session->d_results, &session->resultCapacityElements, neededOutputElements);

    // --- 2. Transferencia de Estado del Batch (Smart Fetch Inputs) ---
    // Copiamos el chorizo comprimido de inputs
    CUDA_CHECK_M(cudaMemcpy(session->d_newInflows, h_newInflows, batchSize * sizeof(float), cudaMemcpyHostToDevice));

    // Copiamos el estado inicial del río (Q y H en t=0) a los buffers persistentes
    CUDA_CHECK_M(cudaMemcpy(session->d_initialQ, h_initialQ, cellBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_M(cudaMemcpy(session->d_initialDepths, h_initialDepths, cellBytes, cudaMemcpyHostToDevice));

    // --- 3. Ejecución del Kernel (Smart Fetch) ---
    launchManningSmartKernel(
        session->d_results,
        session->d_newInflows,
        session->d_initialQ,
        session->d_initialDepths,
        session->d_bottomWidths,
        session->d_sideSlopes,
        session->d_inv_n,
        session->d_sqrt_slope,
        session->d_pythagoras,
        batchSize,
        cellCount
    );

    CUDA_CHECK_M(cudaGetLastError());
    CUDA_CHECK_M(cudaDeviceSynchronize());

    // --- 4. Recuperación de Resultados (Expandidos) ---
    std::vector<float> host_results(neededOutputElements);
    CUDA_CHECK_M(cudaMemcpy(host_results.data(), session->d_results, neededOutputElements * sizeof(float), cudaMemcpyDeviceToHost));

    return host_results;
}

void destroy_manning_session(ManningSession* session) {
    if (!session) return;

    // Liberar Geometría Invariante
    if (session->d_bottomWidths) cudaFree(session->d_bottomWidths);
    if (session->d_sideSlopes)   cudaFree(session->d_sideSlopes);
    if (session->d_inv_n)        cudaFree(session->d_inv_n);
    if (session->d_sqrt_slope)   cudaFree(session->d_sqrt_slope);
    if (session->d_pythagoras)   cudaFree(session->d_pythagoras);

    // Liberar Buffers de Estado
    if (session->d_initialQ)      cudaFree(session->d_initialQ);
    if (session->d_initialDepths) cudaFree(session->d_initialDepths);

    // Liberar Buffers Adaptativos
    if (session->d_results)      cudaFree(session->d_results);
    if (session->d_newInflows)   cudaFree(session->d_newInflows);

    delete session;
}