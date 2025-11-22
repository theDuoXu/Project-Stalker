#include "projectstalker/physics/transport_kernel.h"
#include <cuda_runtime.h>
#include <cmath>

// --- CONSTANTES DE COMPILACIÓN ---
#define BLOCK_SIZE 256
#define HALO_SIZE 2
#define S_MEM_SIZE (BLOCK_SIZE + 2 * HALO_SIZE)

// --- CONSTANTES FÍSICAS (Arrhenius Optimizado) ---
#define ARRHENIUS_REF_T 20.0f
// Pre-calculamos log2(1.047) para usar exp2f que es más rápido que powf
// log2(1.047) ≈ 0.066242226f
#define LOG2_THETA 0.066242226f

// --- FUNCIONES DEVICE ---

__device__ inline float device_minmod(float a, float b) {
    float sign = copysignf(1.0f, a);
    // Branchless check: devuelve 0 si signos opuestos
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
    const float* __restrict__ d_temperature,
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

    // 2. Memoria Compartida
    __shared__ float s_C[S_MEM_SIZE];
    __shared__ float s_U[S_MEM_SIZE];
    __shared__ float s_A[S_MEM_SIZE];

    // 3. CARGA COOPERATIVA (Global -> Shared)
    // --------------------------------------------------

    // Registro temporal para cachear lectura global
    float load_c = 0.0f, load_u = 0.0f, load_a = 0.0f;
    bool inside = (gid < cellCount);

    if (inside) {
        // Lectura coalescente perfecta (LDG cache enabled by __restrict__)
        load_c = d_c_old[gid];
        load_u = d_velocity[gid];
        load_a = d_area[gid];
    }

    // A. Cargar Zona Central
    s_C[s_idx] = load_c;
    s_U[s_idx] = load_u;
    s_A[s_idx] = load_a;

    // B. Cargar Halo Izquierdo
    if (tid < HALO_SIZE) {
        int halo_gid = gid - HALO_SIZE;
        // Boundary Clamp (Copia valor de 0 si estamos fuera) -> Pendiente 0
        int src_gid = (halo_gid >= 0) ? halo_gid : 0;

        s_C[tid] = d_c_old[src_gid];
        s_U[tid] = d_velocity[src_gid];
        s_A[tid] = d_area[src_gid];
    }

    // C. Cargar Halo Derecho
    if (tid >= BLOCK_SIZE - HALO_SIZE) {
        int halo_gid = gid + HALO_SIZE;
        int s_halo_idx = s_idx + HALO_SIZE;
        int src_gid = (halo_gid < cellCount) ? halo_gid : (cellCount - 1);

        s_C[s_halo_idx] = d_c_old[src_gid];
        s_U[s_halo_idx] = d_velocity[src_gid];
        s_A[s_halo_idx] = d_area[src_gid];
    }

    __syncthreads(); // Sincronización Obligatoria

    // 4. CÁLCULO FÍSICO
    // --------------------------------------------------
    if (inside) {

        // --- ADVECCIÓN (MUSCL 2nd Order) ---

        // Lectura ultrarrápida de Shared Memory (L1)
        float c_prev = s_C[s_idx - 1];
        float c_curr = s_C[s_idx];
        float c_next = s_C[s_idx + 1];

        // Cálculo de pendientes
        float r_curr = device_minmod(c_curr - c_prev, c_next - c_curr);
        float r_prev = device_minmod(c_prev - s_C[s_idx - 2], c_curr - c_prev);

        // Reconstrucción
        float c_face_left  = c_prev + 0.5f * r_prev;
        float c_face_right = c_curr + 0.5f * r_curr;

        // Flujos (Usando U y A cacheados en Shared Mem)
        float flux_out = s_U[s_idx] * s_A[s_idx] * c_face_right;
        float flux_in  = s_U[s_idx - 1] * s_A[s_idx - 1] * c_face_left;

        if (gid == 0) flux_in = 0.0f; // Inlet Dirichlet = 0


        // --- DIFUSIÓN (Central Differences) ---

        float h = d_depth[gid];
        float alpha = d_alpha[gid];
        // Branchless max para evitar DL=0
        float DL = fmaxf(alpha * fabsf(load_u) * h, 1e-6f);

        float diff_term = (c_next - 2.0f * c_curr + c_prev);
        float diffusion_change = (DL * diff_term) / (dx * dx);


        // --- REACCIÓN (Arrhenius Optimizado) ---

        float k20 = d_decay[gid];
        float temp = d_temperature[gid];

        // Usamos exp2f en lugar de powf
        // theta^(T-20) = 2^((T-20) * log2(theta))
        float correction = exp2f((temp - ARRHENIUS_REF_T) * LOG2_THETA);
        float k_real = k20 * correction;

        float reaction_change = -k_real * c_curr;


        // --- INTEGRACIÓN TEMPORAL ---

        float vol = load_a * dx;
        // Branchless safe division
        float advection_change = (flux_in - flux_out) / fmaxf(vol, 1e-6f);

        float c_new = c_curr + dt * (advection_change + diffusion_change + reaction_change);

        // Clamp a 0 (No masa negativa)
        d_c_new[gid] = fmaxf(0.0f, c_new);
    }
}

// --- LAUNCHER ---

void launchTransportKernel(
    float* d_c_new, const float* d_c_old,
    const float* d_velocity, const float* d_depth, const float* d_area,
    const float* d_temperature,
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