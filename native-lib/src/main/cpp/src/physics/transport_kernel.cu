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

// -----------------------------------------------------------------------------
// KERNEL 1: BAKING (PRE-COCINADO DE FÍSICA)
// -----------------------------------------------------------------------------
// Se ejecuta UNA VEZ al principio. Prepara los coeficientes para no recalcularlos
// millones de veces en el bucle temporal.
__global__ void bakePhysicsKernel(
    float* __restrict__ d_flow,      // Salida: Caudal (u * A)
    float* __restrict__ d_diff,      // Salida: Coef Difusión (alpha * |u| * h)
    float* __restrict__ d_react,     // Salida: Tasa Reacción (k20 * Arrhenius)
    const float* __restrict__ u,
    const float* __restrict__ h,
    const float* __restrict__ A,
    const float* __restrict__ T,
    const float* __restrict__ alpha,
    const float* __restrict__ k20,
    int cellCount
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < cellCount) {
        // 1. Pre-cálculo de Flujo (Q = u * A)
        // Esto ahorra multiplicaciones en la advección
        d_flow[gid] = u[gid] * A[gid];

        // 2. Pre-cálculo de Difusión (DL = alpha * |u| * h)
        // Fusiona 3 lecturas de memoria en 1 sola variable para el Solver
        d_diff[gid] = alpha[gid] * fabsf(u[gid]) * h[gid];

        // 3. Pre-cálculo de Reacción (Arrhenius)
        // Elimina el cálculo costoso de pow/exp dentro del bucle temporal
        float arrhenius = exp2f((T[gid] - ARRHENIUS_REF_T) * LOG2_THETA);
        d_react[gid] = k20[gid] * arrhenius;
    }
}

// -----------------------------------------------------------------------------
// KERNEL 2: KERNEL PRINCIPAL DE TRANSPORTE (OPTIMIZADO)
// -----------------------------------------------------------------------------

__global__ void
__launch_bounds__(BLOCK_SIZE) // Ayuda al compilador a gestionar registros para maximizar ocupación
transportMusclKernel(
    float* __restrict__ d_c_new,       // Salida
    const float* __restrict__ d_c_old, // Entrada (Solo lectura)
    const float* __restrict__ d_flow,       // <-- INPUT PRE-COCINADO (Q = u*A)
    const float* __restrict__ d_diff_coeff, // <-- INPUT PRE-COCINADO (DL)
    const float* __restrict__ d_react_rate, // <-- INPUT PRE-COCINADO (K_eff)
    const float* __restrict__ d_area,       // Necesario para volumen
    float dx,
    float dt,
    int cellCount
) {
    // 1. Coordenadas
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;
    const int s_idx = tid + HALO_SIZE; // Índice desplazado en Shared Memory

    // 2. Memoria Compartida (L1 Cache manual)
    // OPTIMIZACIÓN: Antes s_U y s_A. Ahora solo s_Q. Ahorramos memoria compartida.
    __shared__ float s_C[S_MEM_SIZE];
    __shared__ float s_Q[S_MEM_SIZE];

    // 3. CARGA DE DATOS (Global -> Shared)
    // Usamos registros temporales para optimizar
    float r_c = 0.0f, r_q = 0.0f;
    const bool inside = (gid < cellCount);

    if (inside) {
        // Carga coalescente (lectura en ráfaga)
        r_c = d_c_old[gid];
        r_q = d_flow[gid]; // Cargamos Q (u*A) directamente
    }

    // Rellenar zona central
    s_C[s_idx] = r_c;
    s_Q[s_idx] = r_q;

    // Rellenar Halo Izquierdo
    if (tid < HALO_SIZE) {
        int halo_gid = gid - HALO_SIZE;
        // Boundary Condition: Clamp (copiar valor del borde 0 si nos salimos)
        int src_gid = (halo_gid >= 0) ? halo_gid : 0;

        s_C[tid] = d_c_old[src_gid];
        s_Q[tid] = d_flow[src_gid];
    }

    // Rellenar Halo Derecho
    if (tid >= BLOCK_SIZE - HALO_SIZE) {
        int halo_gid = gid + HALO_SIZE;
        int s_halo_idx = s_idx + HALO_SIZE;
        // Boundary Condition: Clamp (copiar valor del último borde)
        int src_gid = (halo_gid < cellCount) ? halo_gid : (cellCount - 1);

        s_C[s_halo_idx] = d_c_old[src_gid];
        s_Q[s_halo_idx] = d_flow[src_gid];
    }

    // Barrera: Esperar a que el bloque haya cargado la memoria compartida
    __syncthreads();

    // 4. CÁLCULO FÍSICO (Solo hilos válidos)
    if (inside) {

        // --- LATENCY HIDING MEJORADO (PRE-FETCHING) ---
        // Solicitamos los datos cocinados AHORA.
        // OPTIMIZACIÓN: Solo leemos 3 variables globales en lugar de 6.
        // Menos tráfico en el bus VRAM -> Menos "L1TEX Stalls".
        const float l_diff = d_diff_coeff[gid];
        const float l_k    = d_react_rate[gid];
        const float l_area = d_area[gid]; // Necesario para vol = A * dx

        // --- A. ADVECCIÓN (MUSCL de 2º Orden) ---
        // (Este bloque se ejecuta mientras llegan los datos de arriba)

        float c_LL = s_C[s_idx - 2]; // Left-Left
        float c_L  = s_C[s_idx - 1]; // Left
        float c_C  = s_C[s_idx];     // Center
        float c_R  = s_C[s_idx + 1]; // Right

        // Cálculo de pendientes limitadas
        float slope_L = device_minmod(c_L - c_LL, c_C - c_L);
        float slope_C = device_minmod(c_C - c_L, c_R - c_C);

        // Reconstrucción de valores en las caras de la celda
        // Uso de FMA (Fused Multiply-Add) -> fmaf(a,b,c) = a*b+c
        float c_face_L = fmaf(0.5f, slope_L, c_L);
        float c_face_R = fmaf(0.5f, slope_C, c_C);

        // Cálculo de Flujos (Q = v * A * C)
        // OPTIMIZACIÓN: Usamos s_Q directamente. Ahorramos multiplicaciones u*A.
        float flux_out = s_Q[s_idx]     * c_face_R;
        float flux_in  = s_Q[s_idx - 1] * c_face_L;

        // Condición de Borde en la entrada (Inlet)
        if (gid == 0) flux_in = 0.0f;


        // --- B. DIFUSIÓN (Diferencias Centradas) ---
        // Aquí usamos las variables pre-cocinadas (l_*) que solicitamos al principio.
        // A estas alturas, los datos ya deberían estar disponibles sin espera.

        // Usamos el coeficiente pre-cocinado l_diff (que ya contiene alpha*u*h)
        float DL = fmaxf(l_diff, EPSILON);

        float laplacian = (c_R - 2.0f * c_C + c_L);
        float diffusion_term = (DL * laplacian) / (dx * dx);


        // --- INTEGRACIÓN PARCIAL (Transporte) ---
        float vol = l_area * dx;
        float advection_term = (flux_in - flux_out) / fmaxf(vol, EPSILON);

        // C_star es la concentración después de moverse, pero antes de reaccionar
        // Usamos FMA para el paso de tiempo: c + dt * term
        float c_star = fmaxf(0.0f, fmaf(dt, (advection_term + diffusion_term), c_C));


        // --- C. REACCIÓN (Solución Analítica Exacta) ---
        // Resolvemos dC/dt = -kC  =>  C(t+dt) = C(t) * exp(-k*dt)

        // Usamos el k_eff pre-cocinado (l_k) que ya incluye Arrhenius
        // exp2f(-k * dt * log2e)
        float decay_factor = exp2f(-l_k * dt * LOG2_E);

        // Aplicamos el decaimiento al estado transportado
        float c_final = c_star * decay_factor;

        // 5. ESCRITURA A MEMORIA GLOBAL
        d_c_new[gid] = c_final;
    }
}

// --- LAUNCHERS (Host) ---

void launchBakingKernel(
    float* d_flow, float* d_diff, float* d_react,
    const float* u, const float* h, const float* A,
    const float* T, const float* alpha, const float* k20,
    int cellCount
) {
    int grid = (cellCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    bakePhysicsKernel<<<grid, BLOCK_SIZE>>>(
        d_flow, d_diff, d_react,
        u, h, A, T, alpha, k20,
        cellCount
    );
}

void launchTransportKernel(
    float* d_c_new, const float* d_c_old,
    const float* d_flow, const float* d_diff, const float* d_react, const float* d_area,
    float dx, float dt, int cellCount
) {
    int threadsPerBlock = BLOCK_SIZE;
    // Cálculo estándar de Grid size (ceiling division)
    int blocksPerGrid = (cellCount + threadsPerBlock - 1) / threadsPerBlock;

    transportMusclKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_c_new, d_c_old,
        d_flow, d_diff, d_react, d_area,
        dx, dt, cellCount
    );
}