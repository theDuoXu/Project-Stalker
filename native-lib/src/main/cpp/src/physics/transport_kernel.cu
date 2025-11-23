#include "projectstalker/physics/transport_kernel.h"
#include <cuda_runtime.h>
#include <cmath>

// --- CONSTANTES DE COMPILACIÓN ---
#define BLOCK_SIZE 256
#define HALO_SIZE 2
// Padding para evitar bank conflicts en shared memory (opcional pero recomendado)
#define S_MEM_SIZE (BLOCK_SIZE + 2 * HALO_SIZE)

// --- CONSTANTES MATEMÁTICAS PRE-CALCULADAS ---
#define ARRHENIUS_REF_T 20.0f
// log2(1.047) para la corrección de temperatura theta
#define LOG2_THETA 0.066242226f
// log2(e) para convertir la base natural a base 2
#define LOG2_E 1.44269504f
// Pequeño epsilon para evitar divisiones por cero
#define EPSILON 1e-12f

// --- FUNCIONES AUXILIARES (DEVICE) ---

// MinMod Limiter para evitar oscilaciones en el esquema MUSCL
__device__ inline float device_minmod(float a, float b) {
    // Si tienen signos opuestos, la pendiente es 0 (evita crear nuevos extremos)
    if (a * b <= 0.0f) return 0.0f;
    // Si tienen el mismo signo, tomamos el más pequeño en magnitud
    return copysignf(1.0f, a) * fminf(fabsf(a), fabsf(b));
}

// --- KERNEL PRINCIPAL ---

__global__ void transportMusclKernel(
    float* __restrict__ d_c_new,       // Salida
    const float* __restrict__ d_c_old, // Entrada (Solo lectura)
    const float* __restrict__ d_velocity,
    const float* __restrict__ d_depth,
    const float* __restrict__ d_area,
    const float* __restrict__ d_temperature,
    const float* __restrict__ d_alpha,
    const float* __restrict__ d_decay, // k20
    float dx,
    float dt,
    int cellCount
) {
    // 1. Coordenadas
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;
    const int s_idx = tid + HALO_SIZE; // Índice desplazado en Shared Memory

    // 2. Memoria Compartida (L1 Cache manual)
    __shared__ float s_C[S_MEM_SIZE];
    __shared__ float s_U[S_MEM_SIZE];
    __shared__ float s_A[S_MEM_SIZE];

    // 3. CARGA DE DATOS (Global -> Shared)
    // Usamos registros temporales para optimizar
    float r_c = 0.0f, r_u = 0.0f, r_a = 0.0f;
    const bool inside = (gid < cellCount);

    if (inside) {
        // Carga coalescente (lectura en ráfaga)
        r_c = d_c_old[gid];
        r_u = d_velocity[gid];
        r_a = d_area[gid];
    }

    // Rellenar zona central
    s_C[s_idx] = r_c;
    s_U[s_idx] = r_u;
    s_A[s_idx] = r_a;

    // Rellenar Halo Izquierdo
    if (tid < HALO_SIZE) {
        int halo_gid = gid - HALO_SIZE;
        // Boundary Condition: Clamp (copiar valor del borde 0 si nos salimos)
        int src_gid = (halo_gid >= 0) ? halo_gid : 0;

        s_C[tid] = d_c_old[src_gid];
        s_U[tid] = d_velocity[src_gid];
        s_A[tid] = d_area[src_gid];
    }

    // Rellenar Halo Derecho
    if (tid >= BLOCK_SIZE - HALO_SIZE) {
        int halo_gid = gid + HALO_SIZE;
        int s_halo_idx = s_idx + HALO_SIZE;
        // Boundary Condition: Clamp (copiar valor del último borde)
        int src_gid = (halo_gid < cellCount) ? halo_gid : (cellCount - 1);

        s_C[s_halo_idx] = d_c_old[src_gid];
        s_U[s_halo_idx] = d_velocity[src_gid];
        s_A[s_halo_idx] = d_area[src_gid];
    }

    // Barrera: Esperar a que todo el bloque haya cargado la memoria
    __syncthreads();

    // 4. CÁLCULO FÍSICO (Solo hilos válidos)
    if (inside) {

        // --- A. ADVECCIÓN (MUSCL de 2º Orden) ---
        // Leemos vecinos desde la Shared Memory (muy rápido)
        float c_LL = s_C[s_idx - 2]; // Left-Left (necesario para pendiente previa)
        float c_L  = s_C[s_idx - 1]; // Left
        float c_C  = s_C[s_idx];     // Center
        float c_R  = s_C[s_idx + 1]; // Right

        // Cálculo de pendientes limitadas
        float slope_L = device_minmod(c_L - c_LL, c_C - c_L);
        float slope_C = device_minmod(c_C - c_L, c_R - c_C);

        // Reconstrucción de valores en las caras de la celda
        float c_face_L = c_L + 0.5f * slope_L;
        float c_face_R = c_C + 0.5f * slope_C;

        // Cálculo de Flujos (Q = v * A * C)
        // Nota: Usamos Upwind implícito asumiendo velocidad positiva hacia derecha.
        // Si la velocidad fuera negativa, habría que elegir c_face opuesto.
        float flux_out = s_U[s_idx]     * s_A[s_idx]     * c_face_R;
        float flux_in  = s_U[s_idx - 1] * s_A[s_idx - 1] * c_face_L;

        // Condición de Borde en la entrada (Inlet)
        if (gid == 0) flux_in = 0.0f;


        // --- B. DIFUSIÓN (Diferencias Centradas) ---
        float h = d_depth[gid];
        float alpha = d_alpha[gid];
        // Evitamos DL=0 para no tener problemas numéricos
        float DL = fmaxf(alpha * fabsf(r_u) * h, EPSILON);

        float laplacian = (c_R - 2.0f * c_C + c_L);
        float diffusion_term = (DL * laplacian) / (dx * dx);


        // --- INTEGRACIÓN PARCIAL (Transporte) ---
        float vol = r_a * dx;
        float advection_term = (flux_in - flux_out) / fmaxf(vol, EPSILON);

        // C_star es la concentración después de moverse, pero antes de reaccionar
        float c_star = c_C + dt * (advection_term + diffusion_term);

        // Saneamiento: No permitimos masa negativa por errores numéricos de advección
        c_star = fmaxf(0.0f, c_star);


        // --- C. REACCIÓN (Solución Analítica Exacta) ---
        // Resolvemos dC/dt = -kC  =>  C(t+dt) = C(t) * exp(-k*dt)

        float k20 = d_decay[gid];
        float temp = d_temperature[gid];

        // 1. Corrección de Arrhenius optimizada con base 2
        // k_real = k20 * theta^(T-20)
        // theta^(X) = 2^(X * log2(theta))
        float arrhenius_factor = exp2f((temp - ARRHENIUS_REF_T) * LOG2_THETA);
        float k_real = k20 * arrhenius_factor;

        // 2. Decaimiento Exponencial
        // exp(-k * dt) = 2^(-k * dt * log2(e))
        // Usamos exp2f porque las GPUs tienen hardware dedicado para potencias de 2
        float decay_factor = exp2f(-k_real * dt * LOG2_E);

        // Aplicamos el decaimiento al estado transportado
        float c_final = c_star * decay_factor;

        // 5. ESCRITURA A MEMORIA GLOBAL
        d_c_new[gid] = c_final;
    }
}

// --- LAUNCHER (Host) ---

void launchTransportKernel(
    float* d_c_new, const float* d_c_old,
    const float* d_velocity, const float* d_depth, const float* d_area,
    const float* d_temperature,
    const float* d_alpha, const float* d_decay,
    float dx, float dt, int cellCount
) {
    int threadsPerBlock = BLOCK_SIZE;
    // Cálculo estándar de Grid size (ceiling division)
    int blocksPerGrid = (cellCount + threadsPerBlock - 1) / threadsPerBlock;

    transportMusclKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_c_new, d_c_old,
        d_velocity, d_depth, d_area,
        d_temperature,
        d_alpha, d_decay,
        dx, dt, cellCount
    );
}