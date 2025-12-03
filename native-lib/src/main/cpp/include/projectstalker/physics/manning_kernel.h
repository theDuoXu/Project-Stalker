// src/main/cpp/include/projectstalker/physics/manning_kernel.h
#pragma once

// Esto le dice a cualquier compilador de C++ (g++ o nvcc)
// que debe usar enlace C (nombres limpios) para esta función.
#ifdef __cplusplus
extern "C" {
#endif

/**
 * Kernel de Pre-cálculo (Baking).
 * Se ejecuta UNA VEZ al inicializar la sesión.
 * Calcula constantes matemáticas costosas (raíces, inversos) para no repetirlas en el bucle principal.
 *
 * @param d_inv_n       Salida: 1.0 / n
 * @param d_sqrt_slope  Salida: sqrt(S)
 * @param d_pythagoras  Salida: sqrt(1 + m^2)
 * @param d_manning     Entrada: Coeficientes de Manning originales
 * @param d_bedSlope    Entrada: Pendientes del fondo originales
 * @param d_sideSlope   Entrada: Taludes laterales originales
 * @param cellCount     Número de celdas
 */
void launchManningBakingKernel(
    float* d_inv_n,
    float* d_sqrt_slope,
    float* d_pythagoras,
    const float* d_manning,
    const float* d_bedSlope,
    const float* d_sideSlope,
    int cellCount
);

/**
 * Lanza el kernel de CUDA "Smart Fetch" (Stateful).
 * Resuelve la ecuación de Manning para un lote expandiendo los datos en la GPU.
 *
 * 1. Recibe 'd_newInflows' (pequeño) y 'd_initialQ' (estado base) en lugar de una matriz gigante.
 * 2. Recibe geometría pre-cocinada (inv_n, sqrt_slope...) en lugar de datos crudos.
 *
 * @param d_results       Salida: Matriz completa [H, V] para todo el batch. Tamaño: batchSize * cellCount * 2
 * @param d_newInflows    Entrada: Caudales entrando al sistema en cada paso t. Tamaño: batchSize
 * @param d_initialQ      Entrada: Estado del caudal en el río en t=0. Tamaño: cellCount
 * @param d_initialDepths Entrada: Estado de la profundidad en t=0 (Semilla para Newton-Raphson). Tamaño: cellCount
 * @param d_bottomWidths  Entrada Geometría: Ancho del fondo (b). Tamaño: cellCount
 * @param d_sideSlopes    Entrada Geometría: Talud lateral (m). Tamaño: cellCount
 * @param d_inv_n         Entrada Geometría (Baked): Inverso de Manning (1/n). Tamaño: cellCount
 * @param d_sqrt_slope    Entrada Geometría (Baked): Raíz de la pendiente. Tamaño: cellCount
 * @param d_pythagoras    Entrada Geometría (Baked): Raíz de pitágoras (P). Tamaño: cellCount
 * @param batchSize       Número de pasos de tiempo a simular.
 * @param cellCount       Número de celdas en el río.
 */
void launchManningSmartKernel(
    float* d_results,
    const float* d_newInflows,
    const float* d_initialQ,
    const float* d_initialDepths,
    const float* d_bottomWidths,
    const float* d_sideSlopes,
    const float* d_inv_n,
    const float* d_sqrt_slope,
    const float* d_pythagoras,
    int batchSize,
    int cellCount
);

#ifdef __cplusplus
}
#endif