// src/main/cpp/include/projectstalker/physics/manning_solver.h
#pragma once

#include <vector>
#include <cstddef> // Necesario para size_t

// Modos de Ejecución
enum ManningMode {
    MODE_SMART_LAZY = 0,    // Optimizado: Calcula todo, descarga triángulo. (Alertas/Steady Init)
    MODE_FULL_EVOLUTION = 1 // Robusto: Calcula todo, descarga rectángulo con STRIDE. (Ciencia/Unsteady)
};

/**
 * Estructura de Sesión para Manning (Stateful).
 */
struct ManningSession {
    // 1. Geometría Invariante
    float* d_bottomWidths = nullptr;
    float* d_sideSlopes   = nullptr;
    float* d_inv_n        = nullptr;
    float* d_sqrt_slope   = nullptr;
    float* d_pythagoras   = nullptr;

    // 2. Estado Base (Flyweight / Seed)
    float* d_initialQ      = nullptr;
    float* d_initialDepths = nullptr;

    // 3. Buffers de Trabajo
    float* d_results       = nullptr; // Output global (Stride)
    float* d_newInflows    = nullptr; // Input global

    // 4. PING-PONG STATE BUFFERS
    // Almacenan el estado Q completo del río en t y t-1.
    float* d_ping_Q        = nullptr;
    float* d_pong_Q        = nullptr;

    // 5. Metadatos
    size_t resultCapacityElements = 0;
    size_t inputBatchCapacity     = 0;
    int cellCount = 0;
};

// --- Funciones de Ciclo de Vida ---

ManningSession* init_manning_session(
    const float* h_bottomWidths,
    const float* h_sideSlopes,
    const float* h_manningCoeffs,
    const float* h_bedSlopes,
    const float* h_initialDepths,
    const float* h_initialQ,
    int cellCount
);

/**
 * Ejecuta un batch de simulación.
 *
 * @param mode              Estrategia de simulación (SMART vs FULL).
 * @param session           Puntero a la sesión activa.
 * @param h_pinned_inflows  Puntero a memoria PINNED con los caudales nuevos.
 * @param h_pinned_results  Puntero a memoria PINNED donde escribir.
 * - SMART: Triángulo activo [Batch^2 * 2]
 * - FULL:  Rectángulo total reducido [(Batch/Stride) * CellCount * 2]
 * @param batchSize         Número de pasos a simular.
 * @param stride            Factor de submuestreo para Full Evolution (1 = Todo, N = 1 de cada N).
 */
void run_manning_batch_stateful(
    ManningSession* session,
    const float* h_pinned_inflows,
    float* h_pinned_results,
    int batchSize,
    int mode,
    int stride
);

void destroy_manning_session(ManningSession* session);