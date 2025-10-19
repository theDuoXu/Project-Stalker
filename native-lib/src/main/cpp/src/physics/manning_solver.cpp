// manning_solver.cpp

#include "projectstalker/physics/manning_solver.h"
// Forzamos al compilador de C++ a buscar los nombres C (sin mutilar)
// de las funciones declaradas en este cabezal.
extern "C" {
    #include "projectstalker/physics/manning_kernel.h"
}
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

// --- Macro de Comprobación de Errores de CUDA ---
// Envuelve cada llamada a la API de CUDA para una depuración robusta.
#define CUDA_CHECK(err) { \
    cudaError_t err_code = err; \
    if (err_code != cudaSuccess) { \
        std::cerr << "Error CUDA en " << __FILE__ << ":" << __LINE__ \
                  << ": " << cudaGetErrorString(err_code) << std::endl; \
        throw std::runtime_error("Error en la ejecución de CUDA."); \
    } \
}

std::vector<float> solve_manning_batch_cpp(
    const std::vector<float>& initialGuess,
    const std::vector<float>& flatDischarges,
    int batchSize,
    int cellCount,
    const std::vector<float>& bottomWidths,
    const std::vector<float>& sideSlopes,
    const std::vector<float>& manningCoeffs,
    const std::vector<float>& bedSlopes
) {
    // 1. Calcular tamaños
    const int totalThreads = batchSize * cellCount;
    if (totalThreads == 0) {
        return {}; // No hay nada que procesar
    }

    const size_t total_threads_bytes = totalThreads * sizeof(float);
    const size_t cell_count_bytes = cellCount * sizeof(float);
    const size_t results_bytes = totalThreads * 2 * sizeof(float); // 2 por [Depth, Velocity]

    // 2. Declarar punteros de dispositivo (GPU)
    float *d_initialGuess = nullptr, *d_flatDischarges = nullptr, *d_results = nullptr;
    float *d_bottomWidths = nullptr, *d_sideSlopes = nullptr, *d_manningCoeffs = nullptr, *d_bedSlopes = nullptr;

    try {
        // 3. Reservar memoria en la GPU
        CUDA_CHECK(cudaMalloc(&d_initialGuess, total_threads_bytes));
        CUDA_CHECK(cudaMalloc(&d_flatDischarges, total_threads_bytes));
        CUDA_CHECK(cudaMalloc(&d_bottomWidths, cell_count_bytes));
        CUDA_CHECK(cudaMalloc(&d_sideSlopes, cell_count_bytes));
        CUDA_CHECK(cudaMalloc(&d_manningCoeffs, cell_count_bytes));
        CUDA_CHECK(cudaMalloc(&d_bedSlopes, cell_count_bytes));
        CUDA_CHECK(cudaMalloc(&d_results, results_bytes));

        // 4. Copiar datos desde el Host (CPU) al Device (GPU)
        CUDA_CHECK(cudaMemcpy(d_initialGuess, initialGuess.data(), total_threads_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_flatDischarges, flatDischarges.data(), total_threads_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bottomWidths, bottomWidths.data(), cell_count_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sideSlopes, sideSlopes.data(), cell_count_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_manningCoeffs, manningCoeffs.data(), cell_count_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bedSlopes, bedSlopes.data(), cell_count_bytes, cudaMemcpyHostToDevice));

        // 5. Lanzar el kernel de CUDA
        launchManningKernel(
            d_results, d_initialGuess, d_flatDischarges,
            d_bottomWidths, d_sideSlopes, d_manningCoeffs, d_bedSlopes,
            batchSize, cellCount
        );

        // Comprobar si hubo algún error durante el lanzamiento del kernel
        CUDA_CHECK(cudaGetLastError());

        // 6. Esperar a que todos los hilos de la GPU terminen
        CUDA_CHECK(cudaDeviceSynchronize());

        // 7. Copiar resultados desde el Device (GPU) de vuelta al Host (CPU)
        std::vector<float> host_results(totalThreads * 2);
        CUDA_CHECK(cudaMemcpy(host_results.data(), d_results, results_bytes, cudaMemcpyDeviceToHost));

        // 8. Liberar la memoria de la GPU
        cudaFree(d_initialGuess);
        cudaFree(d_flatDischarges);
        cudaFree(d_results);
        cudaFree(d_bottomWidths);
        cudaFree(d_sideSlopes);
        cudaFree(d_manningCoeffs);
        cudaFree(d_bedSlopes);

        return host_results;

    } catch (const std::exception& e) {
        // En caso de error, asegurarse de liberar toda la memoria que se haya podido reservar
        cudaFree(d_initialGuess);
        cudaFree(d_flatDischarges);
        cudaFree(d_results);
        cudaFree(d_bottomWidths);
        cudaFree(d_sideSlopes);
        cudaFree(d_manningCoeffs);
        cudaFree(d_bedSlopes);
        // Re-lanzar la excepción para que JNI pueda manejarla
        throw;
    }
}