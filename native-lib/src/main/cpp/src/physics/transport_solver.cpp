#include "projectstalker/physics/transport_solver.h"
#include "projectstalker/physics/transport_kernel.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sstream>

// --- Macro de Comprobación de Errores de CUDA ---
#define CUDA_CHECK_T(call) { \
    cudaError_t err_code = call; \
    if (err_code != cudaSuccess) { \
        std::stringstream ss; \
        ss << "CUDA Error en Transport Solver " << __FILE__ << ":" << __LINE__ \
           << " | Código: " << err_code \
           << " | Mensaje: " << cudaGetErrorString(err_code); \
        std::cerr << ss.str() << std::endl; \
        throw std::runtime_error(ss.str()); \
    } \
}

// --- RAII WRAPPER ---
struct CudaBuffer {
    float* ptr = nullptr;

    CudaBuffer() = default;

    // Destructor: Libera memoria automáticamente al salir del scope
    ~CudaBuffer() {
        if (ptr) cudaFree(ptr);
    }

    // Deshabilitar copia (Evita Double Free)
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    // --- Semántica de Movimiento (Move Semantics) ---
    // Permite transferir la propiedad del puntero de forma segura

    // 1. Constructor de movimiento
    CudaBuffer(CudaBuffer&& other) noexcept : ptr(other.ptr) {
        other.ptr = nullptr; // IMPORTANTE: Vaciar al otro
    }

    // 2. Asignación de movimiento
    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr) cudaFree(ptr); // Liberar recurso actual si existe
            ptr = other.ptr;        // Robar puntero
            other.ptr = nullptr;    // Vaciar al otro
        }
        return *this;
    }

    // Helpers de acceso
    float** addr() { return &ptr; }      // Para cudaMalloc(&ptr)
    float* get() const { return ptr; }   // Acceso explícito
    operator float*() const { return ptr; } // Conversión implícita a float*
};

// --- IMPLEMENTACIÓN DEL SOLVER ---

std::vector<float> solve_transport_evolution_cpp(
    const float* h_concentration_in,
    const float* h_velocity,
    const float* h_depth,
    const float* h_area,
    const float* h_temperature,
    const float* h_alpha,
    const float* h_decay,
    float dx,
    float dt_sub,
    int num_steps,
    int cellCount
) {
    // 1. Validaciones
    if (cellCount <= 0 || num_steps <= 0) return {};

    const size_t bytes = cellCount * sizeof(float);

    // 2. Declaración RAII (Gestión automática de memoria)
    CudaBuffer d_c_A, d_c_B;
    CudaBuffer d_u, d_h, d_A, d_T, d_alpha, d_decay;

    // 3. Reserva de Memoria
    CUDA_CHECK_T(cudaMalloc(d_c_A.addr(), bytes));
    CUDA_CHECK_T(cudaMalloc(d_c_B.addr(), bytes)); // Buffer secundario (Pong)

    CUDA_CHECK_T(cudaMalloc(d_u.addr(), bytes));
    CUDA_CHECK_T(cudaMalloc(d_h.addr(), bytes));
    CUDA_CHECK_T(cudaMalloc(d_A.addr(), bytes));
    CUDA_CHECK_T(cudaMalloc(d_T.addr(), bytes));
    CUDA_CHECK_T(cudaMalloc(d_alpha.addr(), bytes));
    CUDA_CHECK_T(cudaMalloc(d_decay.addr(), bytes));

    // --- LIMPIEZA PREVENTIVA ---
    // Inicializamos d_c_B a cero para evitar basura en memoria
    CUDA_CHECK_T(cudaMemset(d_c_B, 0, bytes));

    // 4. Copia de Datos (Host -> Device)
    CUDA_CHECK_T(cudaMemcpy(d_c_A, h_concentration_in, bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK_T(cudaMemcpy(d_u, h_velocity, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_T(cudaMemcpy(d_h, h_depth, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_T(cudaMemcpy(d_A, h_area, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_T(cudaMemcpy(d_T, h_temperature, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_T(cudaMemcpy(d_alpha, h_alpha, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_T(cudaMemcpy(d_decay, h_decay, bytes, cudaMemcpyHostToDevice));

    // 5. Bucle de Sub-stepping (Ping-Pong)
    // Usamos punteros raw para el intercambio, los objetos CudaBuffer retienen la propiedad
    float* d_current = d_c_A;
    float* d_next    = d_c_B;

    for (int step = 0; step < num_steps; step++) {
        // Lanzar Kernel
        launchTransportKernel(
            d_next,    // Output (Futuro)
            d_current, // Input (Pasado)
            d_u, d_h, d_A, d_T, d_alpha, d_decay,
            dx, dt_sub, cellCount
        );

        // Swap de punteros (El futuro ahora es el pasado)
        float* temp = d_current;
        d_current = d_next;
        d_next = temp;
    }

    // Sincronización y chequeo de errores
    CUDA_CHECK_T(cudaGetLastError());
    CUDA_CHECK_T(cudaDeviceSynchronize());

    // 6. Copiar Resultados
    std::vector<float> result(cellCount);
    // Copiamos desde d_current, que contiene el último paso válido tras el swap final
    CUDA_CHECK_T(cudaMemcpy(result.data(), d_current, bytes, cudaMemcpyDeviceToHost));

    return result;

    // Al salir, los destructores ~CudaBuffer() liberan toda la VRAM automáticamente.
}