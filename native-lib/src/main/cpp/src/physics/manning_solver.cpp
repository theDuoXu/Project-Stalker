#include "projectstalker/physics/manning_solver.h"
#include "projectstalker/physics/manning_kernel.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sstream>

// Macro de Chequeo
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

// --- RAII WRAPPER (Reutilizado para seguridad y limpieza automática) ---
struct CudaBufferM {
    float* ptr = nullptr;

    CudaBufferM() = default;
    ~CudaBufferM() { if (ptr) cudaFree(ptr); }

    // No-Copy
    CudaBufferM(const CudaBufferM&) = delete;
    CudaBufferM& operator=(const CudaBufferM&) = delete;

    // Helpers
    float** addr() { return &ptr; }
    operator float*() const { return ptr; }
};

std::vector<float> solve_manning_batch_cpp(
    const float* h_initialGuess,
    const float* h_flatDischarges,
    int batchSize,
    int cellCount,
    const float* h_bottomWidths,
    const float* h_sideSlopes,
    const float* h_manningCoeffs,
    const float* h_bedSlopes
) {
    int totalThreads = batchSize * cellCount;
    if (totalThreads == 0) return {};

    size_t threadBytes = totalThreads * sizeof(float);
    size_t cellBytes = cellCount * sizeof(float);
    size_t resultBytes = totalThreads * 2 * sizeof(float);

    // 1. Declaración RAII (Se limpian solas al salir)
    CudaBufferM d_guess, d_discharge;
    CudaBufferM d_width, d_slope, d_manning, d_bed;
    CudaBufferM d_results;

    // 2. Reserva VRAM
    CUDA_CHECK_M(cudaMalloc(d_guess.addr(), threadBytes));
    CUDA_CHECK_M(cudaMalloc(d_discharge.addr(), threadBytes));

    CUDA_CHECK_M(cudaMalloc(d_width.addr(), cellBytes));
    CUDA_CHECK_M(cudaMalloc(d_slope.addr(), cellBytes));
    CUDA_CHECK_M(cudaMalloc(d_manning.addr(), cellBytes));
    CUDA_CHECK_M(cudaMalloc(d_bed.addr(), cellBytes));

    CUDA_CHECK_M(cudaMalloc(d_results.addr(), resultBytes));

    // 3. Copia Host -> Device (Usando los punteros crudos directos)
    CUDA_CHECK_M(cudaMemcpy(d_guess, h_initialGuess, threadBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_M(cudaMemcpy(d_discharge, h_flatDischarges, threadBytes, cudaMemcpyHostToDevice));

    CUDA_CHECK_M(cudaMemcpy(d_width, h_bottomWidths, cellBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_M(cudaMemcpy(d_slope, h_sideSlopes, cellBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_M(cudaMemcpy(d_manning, h_manningCoeffs, cellBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_M(cudaMemcpy(d_bed, h_bedSlopes, cellBytes, cudaMemcpyHostToDevice));

    // 4. Ejecución Kernel
    launchManningKernel(
        d_results, d_guess, d_discharge,
        d_width, d_slope, d_manning, d_bed,
        batchSize, cellCount
    );

    CUDA_CHECK_M(cudaGetLastError());
    CUDA_CHECK_M(cudaDeviceSynchronize());

    // 5. Copia Vuelta
    std::vector<float> host_results(totalThreads * 2);
    CUDA_CHECK_M(cudaMemcpy(host_results.data(), d_results, resultBytes, cudaMemcpyDeviceToHost));

    return host_results;
}