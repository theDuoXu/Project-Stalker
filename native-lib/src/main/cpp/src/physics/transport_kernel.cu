#include "projectstalker/physics/transport_kernel.h"
#include <cuda_runtime.h>
#include <cmath>

// --- CONSTANTES DE COMPILACIÓN ---
#define BLOCK_SIZE 256
#define HALO_SIZE 2
#define S_MEM_SIZE (BLOCK_SIZE + 2 * HALO_SIZE)

// --- CONSTANTES FÍSICAS (Arrhenius) ---
// Coinciden con FirstOrderReactionSolver.java
#define ARRHENIUS_THETA 1.047f
#define ARRHENIUS_REF_T 20.0f

// --- FUNCIONES DEVICE (INLINE) ---

__device__ inline float device_minmod(float a, float b) {
    float sign = copysignf(1.0f, a);
    if (a * b <= 0.0f) return 0.0f;
    return sign * fminf(fabsf(a), fabsf(b));
}

// --- KERNEL PRINCIPAL ---

__global__ void transportMusclKernel(
    float* __restrict__ d_c_new,
    const float* __restrict__ d_c_old,
    const float* __restrict__ d_velocity,
    const float* __restrict__ d_depth,
    const float* __restrict__ d_area,
    const float* __restrict__ d_temperature, // <--- INPUT NUEVO
    const float* __restrict__ d_alpha,
    const float* __restrict__ d_decay,
    float dx,
    float dt,
    int cellCount
) {
    // 1. Identificación
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int s_idx = tid + HALO_SIZE;

    // 2. Memoria Compartida (C, U, A)
    __shared__ float s_C[S_MEM_SIZE];
    __shared__ float s_U[S_MEM_SIZE];
    __shared__ float s_A[S_MEM_SIZE];

    // 3. CARGA DE DATOS (Patrón Cooperativo con Halo)
    // -------------------------------------------------------------------

    float load_c = 0.0f, load_u = 0.0f, load_a = 0.0f;
    bool inside = (gid < cellCount);

    if (inside) {
        load_c = d_c_old[gid];
        load_u = d_velocity[gid];
        load_a = d_area[gid];
    }

    s_C[s_idx] = load_c;
    s_U[s_idx] = load_u;
    s_A[s_idx] = load_a;

    // Carga de Halo Izquierdo
    if (tid < HALO_SIZE) {
        int halo_gid = gid - HALO_SIZE;
        int src_gid = (halo_gid >= 0) ? halo_gid : 0;

        s_C[tid] = d_c_old[src_gid];
        s_U[tid] = d_velocity[src_gid];
        s_A[tid] = d_area[src_gid];
    }

    // Carga de Halo Derecho
    if (tid >= BLOCK_SIZE - HALO_SIZE) {
        int halo_gid = gid + HALO_SIZE;
        int s_halo_idx = s_idx + HALO_SIZE;
        int src_gid = (halo_gid < cellCount) ? halo_gid : (cellCount - 1);

        s_C[s_halo_idx] = d_c_old[src_gid];
        s_U[s_halo_idx] = d_velocity[src_gid];
        s_A[s_halo_idx] = d_area[src_gid];
    }

    __syncthreads();

    // 4. CÁLCULO FÍSICO
    // ------------------------------------------------
    if (inside) {

        // --- A. ADVECCIÓN (MUSCL) ---
        float c_prev = s_C[s_idx - 1];
        float c_curr = s_C[s_idx];
        float c_next = s_C[s_idx + 1];

        // Pendientes Limitadas
        float r_curr = device_minmod(c_curr - c_prev, c_next - c_curr);
        float r_prev = device_minmod(c_prev - s_C[s_idx - 2], c_curr - c_prev);

        // Reconstrucción
        float c_face_left  = c_prev + 0.5f * r_prev;
        float c_face_right = c_curr + 0.5f * r_curr;

        // Flujos Advectivos
        float flux_out = s_U[s_idx] * s_A[s_idx] * c_face_right;
        float flux_in = s_U[s_idx - 1] * s_A[s_idx - 1] * c_face_left;

        if (gid == 0) flux_in = 0.0f;


        // --- B. DIFUSIÓN (Diferencias Centrales) ---
        float h = d_depth[gid];
        float alpha = d_alpha[gid];
        float DL = fmaxf(alpha * fabsf(load_u) * h, 1e-6f);

        float diff_term = (c_next - 2.0f * c_curr + c_prev);
        float diffusion_change = (DL * diff_term) / (dx * dx);


        // --- C. REACCIÓN (Arrhenius) ---

        float k20 = d_decay[gid];     // Coeficiente base
        float temp = d_temperature[gid]; // Leemos temperatura local

        // Corrección: k_real = k20 * theta^(T - 20)
        // powf es una operación SFU, ligeramente costosa pero necesaria.
        float correction = powf(ARRHENIUS_THETA, temp - ARRHENIUS_REF_T);
        float k_real = k20 * correction;

        float reaction_change = -k_real * c_curr;


        // --- D. ACTUALIZACIÓN FINAL ---

        float vol = load_a * dx;
        float advection_change = (flux_in - flux_out) / fmaxf(vol, 1e-6f);

        float c_new = c_curr + dt * (advection_change + diffusion_change + reaction_change);

        // Saneamiento final
        d_c_new[gid] = fmaxf(0.0f, c_new);
    }
}

// --- LAUNCHER ---

void launchTransportKernel(
    float* d_c_new, const float* d_c_old,
    const float* d_velocity, const float* d_depth, const float* d_area,
    const float* d_temperature, // Nuevo parámetro
    const float* d_alpha, const float* d_decay,
    float dx, float dt, int cellCount
) {
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (cellCount + threadsPerBlock - 1) / threadsPerBlock;

    transportMusclKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_c_new, d_c_old,
        d_velocity, d_depth, d_area,
        d_temperature,
        d_alpha, d_decay,
        dx, dt, cellCount
    );
}