// src/main/cpp/include/projectstalker/physics/manning_solver.h
#pragma once

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

    // --- 2. Estado Base (Flyweight Intrinsic) ---
    // Se cargan una vez en init y actúan como el estado base del río (t=0)
    // para propagar la ola en cada batch.
    float* d_initialQ      = nullptr;
    float* d_initialDepths = nullptr;

    // --- 3. Buffers de Trabajo Adaptativos (Crecen bajo demanda) ---
    // Matriz de salida (ahora optimizada con copia triangular)
    float* d_results       = nullptr;

    // Buffer pequeño de entrada comprimida [BatchSize]
    float* d_newInflows    = nullptr;

    // --- 4. Metadatos de Control de Memoria (High-Water Mark) ---
    size_t resultCapacityElements = 0; // Capacidad actual de d_results (floats)
    size_t inputBatchCapacity     = 0; // Capacidad actual de d_newInflows (floats)

    int cellCount = 0; // Dimensión fija del río
};

// --- Funciones de Ciclo de Vida ---

/**
 * Inicializa la sesión en GPU.
 * 1. Reserva memoria para geometría y ESTADO INICIAL.
 * 2. Ejecuta el kernel de "Baking" para pre-calcular constantes.
 * 3. Retorna el puntero a la estructura de sesión.
 *
 * @param h_initialDepths Estado de profundidad base del río.
 * @param h_initialQ      Estado de caudal base del río.
 * @param cellCount       Número de celdas del río.
 * @return Puntero opaco a la sesión (ManningSession*).
 */
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
 * Ejecuta un batch de simulación utilizando la sesión persistente y memoria PINNED.
 *
 * DMA / ZERO-COPY:
 * Ya no devuelve un vector. Escribe directamente en 'h_pinned_results',
 * que es memoria mapeada directamente desde Java (DirectBuffer).
 *
 * @param session           Puntero a la sesión activa.
 * @param h_pinned_inflows  Puntero a memoria PINNED con los caudales nuevos.
 * @param h_pinned_results  Puntero a memoria PINNED donde escribir [BatchSize^2 * 2].
 * @param batchSize         Número de pasos a simular.
 */
void run_manning_batch_stateful(
    ManningSession* session,
    const float* h_pinned_inflows, // Input Zero-Copy
    float* h_pinned_results,       // Output Zero-Copy
    int batchSize
);

/**
 * Destruye la sesión y libera toda la memoria GPU asociada.
 */
void destroy_manning_session(ManningSession* session);