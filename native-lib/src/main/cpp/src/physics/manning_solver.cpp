// src/main/cpp/src/physics/manning_solver.cpp
#include "projectstalker/physics/manning_solver.h"
#include "projectstalker/physics/manning_kernel.h"
#include <cuda_runtime.h>
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
    const float* h_initialDepths, // Estado base (Intrinsic)
    const float* h_initialQ,      // Estado base (Intrinsic)
    int cellCount
) {
    if (cellCount <= 0) return nullptr;

    ManningSession* session = new ManningSession();
    session->cellCount = cellCount;
    size_t cellBytes = cellCount * sizeof(float);

    float* d_temp_manning = nullptr;
    float* d_temp_bed = nullptr;

    try {
        // 1. Reserva VRAM para Geometría Invariante
        CUDA_CHECK_M(cudaMalloc(&session->d_bottomWidths, cellBytes));
        CUDA_CHECK_M(cudaMalloc(&session->d_sideSlopes, cellBytes));
        CUDA_CHECK_M(cudaMalloc(&session->d_inv_n, cellBytes));
        CUDA_CHECK_M(cudaMalloc(&session->d_sqrt_slope, cellBytes));
        CUDA_CHECK_M(cudaMalloc(&session->d_pythagoras, cellBytes));

        // 2. Reserva VRAM para Buffers de Estado del Río (Tamaño Fijo)
        CUDA_CHECK_M(cudaMalloc(&session->d_initialQ, cellBytes));
        CUDA_CHECK_M(cudaMalloc(&session->d_initialDepths, cellBytes));

        // 3. Reserva Temporales para Baking
        CUDA_CHECK_M(cudaMalloc(&d_temp_manning, cellBytes));
        CUDA_CHECK_M(cudaMalloc(&d_temp_bed, cellBytes));

        // 4. Copia Host -> Device (Heavy Lifting - Una sola vez)
        CUDA_CHECK_M(cudaMemcpy(session->d_bottomWidths, h_bottomWidths, cellBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK_M(cudaMemcpy(session->d_sideSlopes, h_sideSlopes, cellBytes, cudaMemcpyHostToDevice));

        // Copiamos el estado inicial una sola vez (Flyweight Intrinsic)
        CUDA_CHECK_M(cudaMemcpy(session->d_initialQ, h_initialQ, cellBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK_M(cudaMemcpy(session->d_initialDepths, h_initialDepths, cellBytes, cudaMemcpyHostToDevice));

        CUDA_CHECK_M(cudaMemcpy(d_temp_manning, h_manningCoeffs, cellBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK_M(cudaMemcpy(d_temp_bed, h_bedSlopes, cellBytes, cudaMemcpyHostToDevice));

        // 5. BAKING: Pre-calcular constantes físicas en GPU
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

        cudaFree(d_temp_manning);
        cudaFree(d_temp_bed);

        return session;

    } catch (...) {
        if (d_temp_manning) cudaFree(d_temp_manning);
        if (d_temp_bed) cudaFree(d_temp_bed);
        destroy_manning_session(session);
        throw;
    }
}

static void resize_buffer_if_needed(float** ptr, size_t* currentCap, size_t needed) {
    if (needed > *currentCap || *currentCap > (needed * 2)) {
        if (*ptr) {
            cudaFree(*ptr);
            *ptr = nullptr;
        }
        size_t newCap = (size_t)(needed * 1.2f);
        if (newCap > 0) {
            if (cudaMalloc(ptr, newCap * sizeof(float)) != cudaSuccess) {
                newCap = needed;
                CUDA_CHECK_M(cudaMalloc(ptr, newCap * sizeof(float)));
            }
        }
        *currentCap = newCap;
    }
}

// DMA / Zero-Copy Enabled Version
void run_manning_batch_stateful(
    ManningSession* session,
    const float* h_pinned_inflows, // DMA Input Pointer
    float* h_pinned_results,       // DMA Output Pointer
    int batchSize
) {
    if (!session || batchSize <= 0) return;

    int cellCount = session->cellCount;
    // El kernel escribe en formato SoA, necesitamos espacio para todo el bloque [H... | V...]
    // Nota: Aunque solo copiamos un subconjunto de vuelta, la GPU calcula todo o escribe en un buffer lineal.
    // Aquí asumimos que d_results debe ser capaz de alojar (BatchSize * CellCount * 2).
    size_t totalThreads = (size_t)batchSize * cellCount;
    size_t neededOutputElements = totalThreads * 2; // H y V

    // --- 1. Gestión de Memoria Adaptativa (Buffers Internos GPU) ---
    resize_buffer_if_needed(&session->d_newInflows, &session->inputBatchCapacity, (size_t)batchSize);
    resize_buffer_if_needed(&session->d_results, &session->resultCapacityElements, neededOutputElements);

    // --- 2. Transferencia Mínima (Input via DMA) ---
    // Al ser h_pinned_inflows memoria pinned (DirectBuffer), el driver usa DMA directo.
    CUDA_CHECK_M(cudaMemcpy(session->d_newInflows, h_pinned_inflows, batchSize * sizeof(float), cudaMemcpyHostToDevice));

    // --- 3. Ejecución del Kernel (Smart Fetch + SoA Output) ---
    launchManningSmartKernel(
        session->d_results,
        session->d_newInflows,
        session->d_initialQ,      // Intrinsic (VRAM)
        session->d_initialDepths, // Intrinsic (VRAM)
        session->d_bottomWidths,
        session->d_sideSlopes,
        session->d_inv_n,
        session->d_sqrt_slope,
        session->d_pythagoras,
        batchSize,
        cellCount
    );

    CUDA_CHECK_M(cudaGetLastError());

    // --- 4. Recuperación de Resultados (DMA Output - Triangular Transfer) ---

    // AHORRO CRÍTICO: No bajamos todo el río (CellCount).
    // Solo bajamos la región donde la ola ha podido llegar.
    int activeWidth = batchSize;
    // Seguridad: Si el batch es mayor que el río (raro), limitamos al río.
    if (activeWidth > cellCount) activeWidth = cellCount;

    // Configuración para cudaMemcpy2D
    // Pitch (ancho de fila en bytes)
    size_t srcPitch = cellCount * sizeof(float);   // Ancho real en GPU
    size_t dstPitch = activeWidth * sizeof(float); // Ancho compactado en CPU (Host Buffer)
    size_t widthBytes = activeWidth * sizeof(float);
    size_t height = batchSize;

    // A. Copiar bloque H (Inicio de d_results) -> Primera mitad de h_pinned_results
    float* src_H = session->d_results;
    float* dst_H = h_pinned_results;

    CUDA_CHECK_M(cudaMemcpy2D(
        dst_H, dstPitch,
        src_H, srcPitch,
        widthBytes, height,
        cudaMemcpyDeviceToHost
    ));

    // B. Copiar bloque V (Desplazado por totalThreads en GPU) -> Segunda mitad de h_pinned_results
    // Calculamos el offset en el buffer de salida: [ Bloque H (Batch x Width) ] [ AQUÍ EMPIEZA V ]
    float* src_V = session->d_results + totalThreads; // Offset SoA en GPU
    float* dst_V = h_pinned_results + (batchSize * activeWidth); // Offset lineal en Host

    CUDA_CHECK_M(cudaMemcpy2D(
        dst_V, dstPitch,
        src_V, srcPitch,
        widthBytes, height,
        cudaMemcpyDeviceToHost
    ));

    // Sincronización final obligatoria para asegurar que la escritura DMA terminó
    // antes de que Java intente leer el buffer.
    CUDA_CHECK_M(cudaDeviceSynchronize());
}

void destroy_manning_session(ManningSession* session) {
    if (!session) return;
    if (session->d_bottomWidths) cudaFree(session->d_bottomWidths);
    if (session->d_sideSlopes)   cudaFree(session->d_sideSlopes);
    if (session->d_inv_n)        cudaFree(session->d_inv_n);
    if (session->d_sqrt_slope)   cudaFree(session->d_sqrt_slope);
    if (session->d_pythagoras)   cudaFree(session->d_pythagoras);
    if (session->d_initialQ)      cudaFree(session->d_initialQ);
    if (session->d_initialDepths) cudaFree(session->d_initialDepths);
    if (session->d_results)      cudaFree(session->d_results);
    if (session->d_newInflows)   cudaFree(session->d_newInflows);
    delete session;
}