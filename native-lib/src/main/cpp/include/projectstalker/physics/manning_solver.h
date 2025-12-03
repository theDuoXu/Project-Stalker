// src/main/cpp/include/projectstalker/physics/manning_solver.h
#pragma once

#include <vector>

/**
 * Estructura de Sesión para Manning (Stateful).
 * Mantiene la geometría residente en VRAM y gestiona buffers adaptativos
 * para evitar asignaciones de memoria (cudaMalloc) en cada frame.
 */
struct ManningSession {
    // --- 1. Geometría Invariante (Se carga una vez en init) ---
    // Punteros crudos necesarios para calcular Área y Perímetro
    float* d_bottomWidths = nullptr;
    float* d_sideSlopes   = nullptr;

    // Constantes físicas pre-cocinadas (Baked) para optimización
    float* d_inv_n        = nullptr; // 1.0 / n
    float* d_sqrt_slope   = nullptr; // sqrt(S)
    float* d_pythagoras   = nullptr; // sqrt(1 + m^2)

    // --- 2. Buffers de Trabajo Adaptativos (Crecen bajo demanda) ---
    // Matriz gigante de salida [BatchSize * CellCount * 2]
    float* d_results       = nullptr;

    // Buffer pequeño de entrada comprimida [BatchSize]
    float* d_newInflows    = nullptr;

    // Estado del río al inicio del batch [CellCount]
    float* d_initialQ      = nullptr;
    float* d_initialDepths = nullptr;

    // --- 3. Metadatos de Control de Memoria (High-Water Mark) ---
    size_t resultCapacityElements = 0; // Capacidad actual de d_results (floats)
    size_t inputBatchCapacity     = 0; // Capacidad actual de d_newInflows (floats)

    int cellCount = 0; // Dimensión fija del río
};

// --- Funciones de Ciclo de Vida ---

/**
 * Inicializa la sesión en GPU.
 * 1. Reserva memoria para la geometría.
 * 2. Ejecuta el kernel de "Baking" para pre-calcular constantes.
 * 3. Retorna el puntero a la estructura de sesión.
 *
 * @param cellCount Número de celdas del río.
 * @return Puntero opaco a la sesión (ManningSession*).
 */
ManningSession* init_manning_session(
    const float* h_bottomWidths,
    const float* h_sideSlopes,
    const float* h_manningCoeffs,
    const float* h_bedSlopes,
    int cellCount
);

/**
 * Ejecuta un batch de simulación utilizando la sesión persistente.
 * Implementa lógica de memoria adaptativa: si el batchSize crece, redimensiona los buffers.
 * Utiliza la lógica "Smart Fetch" para expandir los datos en GPU.
 *
 * @param session         Puntero a la sesión activa.
 * @param h_newInflows    Array comprimido de caudales de entrada [BatchSize].
 * @param h_initialDepths Estado de profundidad en t=0 (Semilla NR) [CellCount].
 * @param h_initialQ      Estado de caudal en t=0 (Base para propagación) [CellCount].
 * @param batchSize       Número de pasos a simular.
 * @return Vector con todos los resultados desplegados [H0, V0, H1, V1...].
 */
std::vector<float> run_manning_batch_stateful(
    ManningSession* session,
    const float* h_newInflows,
    const float* h_initialDepths,
    const float* h_initialQ,
    int batchSize
);

/**
 * Destruye la sesión y libera toda la memoria GPU asociada.
 */
void destroy_manning_session(ManningSession* session);