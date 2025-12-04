// src/main/cpp/src/physics/manning_solver.cpp
#include "projectstalker/physics/manning_solver.h"
#include "projectstalker/physics/manning_kernel.h"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sstream>

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

ManningSession* init_manning_session(
    const float* h_bottomWidths,
    const float* h_sideSlopes,
    const float* h_manningCoeffs,
    const float* h_bedSlopes,
    const float* h_initialDepths,
    const float* h_initialQ,
    int cellCount
) {
    if (cellCount <= 0) return nullptr;

    ManningSession* session = new ManningSession();
    session->cellCount = cellCount;
    size_t cellBytes = cellCount * sizeof(float);

    float* d_temp_manning = nullptr;
    float* d_temp_bed = nullptr;

    try {
        // 0. Crear Stream Asíncrono (Non-Blocking para evitar serialización con CPU)
        CUDA_CHECK_M(cudaStreamCreateWithFlags(&session->stream, cudaStreamNonBlocking));

        // 1. Reserva VRAM (Síncrona, solo ocurre una vez al inicio)
        CUDA_CHECK_M(cudaMalloc(&session->d_bottomWidths, cellBytes));
        CUDA_CHECK_M(cudaMalloc(&session->d_sideSlopes, cellBytes));
        CUDA_CHECK_M(cudaMalloc(&session->d_inv_n, cellBytes));
        CUDA_CHECK_M(cudaMalloc(&session->d_sqrt_slope, cellBytes));
        CUDA_CHECK_M(cudaMalloc(&session->d_pythagoras, cellBytes));
        CUDA_CHECK_M(cudaMalloc(&session->d_initialQ, cellBytes));
        CUDA_CHECK_M(cudaMalloc(&session->d_initialDepths, cellBytes));
        CUDA_CHECK_M(cudaMalloc(&d_temp_manning, cellBytes));
        CUDA_CHECK_M(cudaMalloc(&d_temp_bed, cellBytes));

        // 2. Transferencias Iniciales (Usamos Async aunque esperamos al final para asegurar orden)
        // Nota: Usamos el stream para empezar a encolar trabajo
        CUDA_CHECK_M(cudaMemcpyAsync(session->d_bottomWidths, h_bottomWidths, cellBytes, cudaMemcpyHostToDevice, session->stream));
        CUDA_CHECK_M(cudaMemcpyAsync(session->d_sideSlopes, h_sideSlopes, cellBytes, cudaMemcpyHostToDevice, session->stream));
        CUDA_CHECK_M(cudaMemcpyAsync(session->d_initialQ, h_initialQ, cellBytes, cudaMemcpyHostToDevice, session->stream));
        CUDA_CHECK_M(cudaMemcpyAsync(session->d_initialDepths, h_initialDepths, cellBytes, cudaMemcpyHostToDevice, session->stream));
        CUDA_CHECK_M(cudaMemcpyAsync(d_temp_manning, h_manningCoeffs, cellBytes, cudaMemcpyHostToDevice, session->stream));
        CUDA_CHECK_M(cudaMemcpyAsync(d_temp_bed, h_bedSlopes, cellBytes, cudaMemcpyHostToDevice, session->stream));

        // 3. Kernel Baking (Encolado en el Stream)
        // Nota: launchManningBakingKernel debe modificarse para aceptar stream,
        // pero por ahora usamos el default stream y sincronizamos.
        // *MEJORA PRO*: Sincronizamos el stream antes de llamar al baking si el baking usa stream 0.
        // Para simplificar sin tocar el .cu ahora mismo: Forzamos sync.
        CUDA_CHECK_M(cudaStreamSynchronize(session->stream));

        launchManningBakingKernel(
            session->d_inv_n,
            session->d_sqrt_slope,
            session->d_pythagoras,
            d_temp_manning,
            d_temp_bed,
            session->d_sideSlopes,
            cellCount
        );
        CUDA_CHECK_M(cudaDeviceSynchronize()); // Wait for baking

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

// -----------------------------------------------------------------------------
// RUN BATCH: PIPELINE ASÍNCRONO
// -----------------------------------------------------------------------------
void run_manning_batch_stateful(
    ManningSession* session,
    const float* h_pinned_inflows,
    float* h_pinned_results,
    int batchSize
) {
    if (!session || batchSize <= 0) return;

    int cellCount = session->cellCount;
    size_t totalThreads = (size_t)batchSize * cellCount;
    size_t neededOutputElements = totalThreads * 2;

    // 1. Gestión de Memoria (Síncrona - Rara vez ocurre gracias a la histéresis)
    resize_buffer_if_needed(&session->d_newInflows, &session->inputBatchCapacity, (size_t)batchSize);
    resize_buffer_if_needed(&session->d_results, &session->resultCapacityElements, neededOutputElements);

    // 2. INPUT: Transferencia Asíncrona (HtoD)
    // El bus PCIe sube los datos mientras la CPU prepara la llamada al kernel.
    CUDA_CHECK_M(cudaMemcpyAsync(
        session->d_newInflows,
        h_pinned_inflows,
        batchSize * sizeof(float),
        cudaMemcpyHostToDevice,
        session->stream // <--- STREAM
    ));

    // 3. COMPUTE: Ejecución Asíncrona
    // Lanzamos el kernel al stream. Para una RTX 5090, lanzamos todo de golpe.
    // Partir esto en chunks solo añadiría overhead de CPU launch latency.
    // Nota: Necesitamos actualizar 'launchManningSmartKernel' en el .cu para aceptar stream.
    // COMO NO QUEREMOS TOCAR EL .CU AHORA:
    // El kernel por defecto va al Stream 0 (Legacy).
    // Para que funcione el pipelining con Stream 0, sincronizamos el stream custom antes.
    // **FIX CORRECTO:** Modificar manning_kernel.cu es lo ideal.
    // **WORKAROUND VALID:** Stream 0 serializa con otros streams.

    // Asumiremos que el kernel va al stream por defecto (0).
    // Esperamos a que la copia termine antes de computar (Stream 0 espera a todos implícitamente? No siempre).
    // Lo correcto sin tocar .cu:
    CUDA_CHECK_M(cudaStreamSynchronize(session->stream)); // Espera copia

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

    // 4. OUTPUT: Transferencia Asíncrona (DtoH)
    // El Stream 0 (Kernel) bloqueará el siguiente comando en Stream 0.
    // Pero cudaMemcpyAsync requiere stream específico para no bloquear CPU.
    // Usamos el stream por defecto (0) en el MemcpyAsync para encadenarlo tras el kernel.

    int activeWidth = batchSize;
    if (activeWidth > cellCount) activeWidth = cellCount;

    size_t srcPitch = cellCount * sizeof(float);
    size_t dstPitch = activeWidth * sizeof(float);
    size_t widthBytes = activeWidth * sizeof(float);
    size_t height = batchSize;

    // H
    CUDA_CHECK_M(cudaMemcpy2DAsync(
        h_pinned_results, dstPitch,
        session->d_results, srcPitch,
        widthBytes, height,
        cudaMemcpyDeviceToHost,
        0 // Stream 0 (Default) para asegurar que va después del Kernel
    ));

    // V
    float* src_V = session->d_results + totalThreads;
    float* dst_V = h_pinned_results + (batchSize * activeWidth);

    CUDA_CHECK_M(cudaMemcpy2DAsync(
        dst_V, dstPitch,
        src_V, srcPitch,
        widthBytes, height,
        cudaMemcpyDeviceToHost,
        0 // Stream 0
    ));

    // 5. SYNC FINAL (Barrier)
    // Java necesita los datos YA. Aquí pagamos la latencia.
    CUDA_CHECK_M(cudaStreamSynchronize(0));
}

void destroy_manning_session(ManningSession* session) {
    if (!session) return;

    // Destruir Stream
    if (session->stream) cudaStreamDestroy(session->stream);

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