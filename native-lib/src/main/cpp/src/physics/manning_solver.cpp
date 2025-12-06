// src/main/cpp/src/physics/manning_solver.cpp
#include "projectstalker/physics/manning_solver.h"
#include "projectstalker/physics/manning_kernel.h"
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm> // std::swap
#include <stdexcept>
#include <string>
#include <sstream>

#define CUDA_CHECK_M(call) { \
    cudaError_t err_code = call; \
    if (err_code != cudaSuccess) { \
        std::stringstream ss; \
        ss << "CUDA Error en Manning Solver " << __FILE__ << ":" << __LINE__ \
           << " | Código: " << err_code \
           << " | Mensaje: " << cudaGetErrorString(err_code); \
        std::cerr << ss.str() << std::endl; \
        throw std::runtime_error(ss.str()); \
    } \
}

ManningSession* init_manning_session(
    const float* h_bottomWidths,
    const float* h_sideSlopes,
    const float* h_manningCoeffs,
    const float* h_bedSlopes,
    const float* h_initialDepths,
    const float* h_initialQ,
    int cellCount
) {
    if (cellCount <= 0) return nullptr;

    ManningSession* session = new ManningSession();
    session->cellCount = cellCount;
    session->graphCreated = false;
    session->graphExec = nullptr;
    session->graph = nullptr;

    // Crear Stream Dedicado (Non-Blocking para solapamiento futuro)
    CUDA_CHECK_M(cudaStreamCreateWithFlags(&session->stream, cudaStreamNonBlocking));

    size_t cellBytes = cellCount * sizeof(float);

    float* d_temp_manning = nullptr;
    float* d_temp_bed = nullptr;

    try {
        // 1. Buffers Geometría y Estado Base
        CUDA_CHECK_M(cudaMalloc(&session->d_bottomWidths, cellBytes));
        CUDA_CHECK_M(cudaMalloc(&session->d_sideSlopes, cellBytes));
        CUDA_CHECK_M(cudaMalloc(&session->d_inv_n, cellBytes));
        CUDA_CHECK_M(cudaMalloc(&session->d_sqrt_slope, cellBytes));
        CUDA_CHECK_M(cudaMalloc(&session->d_pythagoras, cellBytes));
        CUDA_CHECK_M(cudaMalloc(&session->d_initialQ, cellBytes));
        CUDA_CHECK_M(cudaMalloc(&session->d_initialDepths, cellBytes));

        // 2. Buffers Ping-Pong (Estado Dinámico)
        CUDA_CHECK_M(cudaMalloc(&session->d_ping_Q, cellBytes));
        CUDA_CHECK_M(cudaMalloc(&session->d_pong_Q, cellBytes));

        // 3. Temporales
        CUDA_CHECK_M(cudaMalloc(&d_temp_manning, cellBytes));
        CUDA_CHECK_M(cudaMalloc(&d_temp_bed, cellBytes));

        // Transferencias (Async en el stream por defecto o dedicado)
        // Usamos sync copy por simplicidad en init
        CUDA_CHECK_M(cudaMemcpy(session->d_bottomWidths, h_bottomWidths, cellBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK_M(cudaMemcpy(session->d_sideSlopes, h_sideSlopes, cellBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK_M(cudaMemcpy(session->d_initialQ, h_initialQ, cellBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK_M(cudaMemcpy(session->d_initialDepths, h_initialDepths, cellBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK_M(cudaMemcpy(d_temp_manning, h_manningCoeffs, cellBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK_M(cudaMemcpy(d_temp_bed, h_bedSlopes, cellBytes, cudaMemcpyHostToDevice));

        // Inicializar Ping-Pong con estado inicial (para que el primer paso tenga datos previos)
        CUDA_CHECK_M(cudaMemcpy(session->d_ping_Q, h_initialQ, cellBytes, cudaMemcpyHostToDevice));
        // Pong no necesita init, se sobrescribirá.

        launchManningBakingKernel(
            session->d_inv_n,
            session->d_sqrt_slope,
            session->d_pythagoras,
            d_temp_manning,
            d_temp_bed,
            session->d_sideSlopes,
            cellCount,
            session->stream // Pasamos el stream
        );
        CUDA_CHECK_M(cudaStreamSynchronize(session->stream));

        cudaFree(d_temp_manning);
        cudaFree(d_temp_bed);

        return session;

    } catch (...) {
        if (d_temp_manning) cudaFree(d_temp_manning);
        if (d_temp_bed) cudaFree(d_temp_bed);
        destroy_manning_session(session);
        throw;
    }
}

static void resize_buffer_if_needed(float** ptr, size_t* currentCap, size_t needed) {
    if (needed > *currentCap || *currentCap > (needed * 2)) {
        if (*ptr) {
            cudaFree(*ptr);
            *ptr = nullptr;
        }
        size_t newCap = (size_t)(needed * 1.2f);
        if (newCap > 0) {
            if (cudaMalloc(ptr, newCap * sizeof(float)) != cudaSuccess) {
                newCap = needed;
                CUDA_CHECK_M(cudaMalloc(ptr, newCap * sizeof(float)));
            }
        }
        *currentCap = newCap;
    }
}

// -----------------------------------------------------------------------------
// HELPER: ESTRATEGIA FULL EVOLUTION (CUDA GRAPHS - Re-Capture Strategy)
// -----------------------------------------------------------------------------
static void run_full_evolution_strategy(
    ManningSession* session,
    const float* h_pinned_inflows,
    float* h_pinned_results,
    int batchSize,
    int stride,
    int savedSteps
) {
    int cellCount = session->cellCount;

    // 1. GESTIÓN DE GRAFOS
    // Optamos por la estrategia de Re-Captura en cada batch para "quemar" los nuevos valores
    // de inflow en el grafo sin complicar la actualización de parámetros del kernel.
    // La captura en CPU es mucho más rápida que el lanzamiento de miles de kernels a la GPU.

    // INICIO CAPTURA DE GRAFO
    // Todo lo que ocurra en el stream a partir de aquí NO se ejecuta, se graba.
    CUDA_CHECK_M(cudaStreamBeginCapture(session->stream, cudaStreamCaptureModeGlobal));

    // Punteros locales para el swap (simulación de ping-pong durante la captura)
    float* d_curr = session->d_ping_Q;
    float* d_prev = session->d_pong_Q;

    // Offset entre bloques H y V en el buffer de salida reducido
    int output_pitch_elements = savedSteps * cellCount;
    int saved_count = 0;

    // Bucle de Orquestación (Grabación)
    for (int t = 0; t < batchSize; t++) {
        // Swap: El que fue Current (escritura) en t-1 pasa a ser Prev (lectura) en t
        std::swap(d_curr, d_prev);

        // Input: Leemos del Host Pinned array. Al capturar, se graba el valor 'float' actual.
        // Esto evita tener que subir los inflows a un buffer de GPU.
        float current_inflow = h_pinned_inflows[t];

        // Stride Logic
        bool save = (t % stride == 0);
        int write_idx_h = -1;
        int write_idx_v = -1;

        if (save) {
            // Calculamos índice plano en el buffer de salida
            write_idx_h = saved_count * cellCount;
            write_idx_v = write_idx_h + output_pitch_elements;
            saved_count++;
        }

        // Lanzar Kernel Paso a Paso (Grabación de Nodo)
        // Se usa el stream de captura.
        launchManningStepKernel(
            d_curr,         // Escribir aquí
            d_prev,         // Leer de aquí
            session->d_results,
            current_inflow, // Se graba el valor literal
            session->d_initialDepths,
            session->d_bottomWidths, session->d_sideSlopes,
            session->d_inv_n, session->d_sqrt_slope, session->d_pythagoras,
            cellCount,
            save,
            write_idx_h,
            write_idx_v,
            session->stream // Stream de sesión
        );
    }

    // Persistencia del Estado (Actualizamos los punteros de sesión para el siguiente batch)
    session->d_ping_Q = d_curr;
    session->d_pong_Q = d_prev;

    // FIN CAPTURA
    cudaGraph_t captured_graph;
    CUDA_CHECK_M(cudaStreamEndCapture(session->stream, &captured_graph));

    // INSTANCIACIÓN (Compilación del Grafo)
    cudaGraphExec_t graphExec;
    CUDA_CHECK_M(cudaGraphInstantiate(&graphExec, captured_graph, NULL, NULL, 0));

    // EJECUCIÓN (Lanzamiento Único)
    // La CPU envía un solo comando. La GPU ejecuta los 'batchSize' pasos sin interrupción.
    CUDA_CHECK_M(cudaGraphLaunch(graphExec, session->stream));

    // LIMPIEZA INMEDIATA (Estrategia Re-Capture)
    // Destruimos el grafo temporal.
    CUDA_CHECK_M(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK_M(cudaGraphDestroy(captured_graph));

    // Output Retrieval (Async en el mismo stream)
    // Esta copia se encola y comenzará inmediatamente después de que termine el grafo.
    size_t bytesToCopy = saved_count * cellCount * 2 * sizeof(float);
    CUDA_CHECK_M(cudaMemcpyAsync(
        h_pinned_results,
        session->d_results,
        bytesToCopy,
        cudaMemcpyDeviceToHost,
        session->stream
    ));
}

// -----------------------------------------------------------------------------
// HELPER: ESTRATEGIA SMART (Monolítica Original)
// -----------------------------------------------------------------------------
static void run_smart_strategy(
    ManningSession* session,
    const float* h_pinned_inflows,
    float* h_pinned_results,
    int batchSize
) {
    int cellCount = session->cellCount;

    // 1. Input Inflows (Array completo)
    // En modo Smart necesitamos subir todo el array de inputs a la GPU de golpe.
    // Usamos Async Copy en el stream de la sesión.
    resize_buffer_if_needed(&session->d_newInflows, &session->inputBatchCapacity, batchSize);
    CUDA_CHECK_M(cudaMemcpyAsync(
        session->d_newInflows,
        h_pinned_inflows,
        batchSize * sizeof(float),
        cudaMemcpyHostToDevice,
        session->stream
    ));

    // 2. Ejecución Kernel
    // Pasamos el stream por defecto (o null implícito) al launcher porque Smart no usa Grafos.
    launchManningSmartKernel(
        session->d_results, session->d_newInflows, session->d_initialQ,
        session->d_initialDepths, session->d_bottomWidths, session->d_sideSlopes,
        session->d_inv_n, session->d_sqrt_slope, session->d_pythagoras,
        batchSize, cellCount
    );

    // 3. Output Retrieval (Triangular 2D Copy Async)
    // Recorte triangular para ahorrar ancho de banda
    int activeWidth = (batchSize > cellCount) ? cellCount : batchSize;

    size_t srcPitch = cellCount * sizeof(float);
    size_t dstPitch = activeWidth * sizeof(float);
    size_t widthBytes = activeWidth * sizeof(float);

    // Copiar H
    CUDA_CHECK_M(cudaMemcpy2DAsync(
        h_pinned_results, dstPitch,
        session->d_results, srcPitch,
        widthBytes, batchSize,
        cudaMemcpyDeviceToHost,
        session->stream
    ));

    // Copiar V (Offset en GPU es Batch*CellCount, en CPU es Batch*ActiveWidth)
    float* src_V = session->d_results + ((size_t)batchSize * cellCount);
    float* dst_V = h_pinned_results + (batchSize * activeWidth);

    CUDA_CHECK_M(cudaMemcpy2DAsync(
        dst_V, dstPitch,
        src_V, srcPitch,
        widthBytes, batchSize,
        cudaMemcpyDeviceToHost,
        session->stream
    ));
}

// -----------------------------------------------------------------------------
// RUN BATCH: ORQUESTADOR PRINCIPAL
// -----------------------------------------------------------------------------
void run_manning_batch_stateful(
    ManningSession* session,
    const float* h_pinned_inflows,
    float* h_pinned_results,
    int batchSize,
    int mode,
    int stride
) {
    if (!session || batchSize <= 0) return;
    int cellCount = session->cellCount;

    // 1. Configuración de Memoria de Salida Global
    // Calculamos cuánto espacio necesitamos en VRAM para guardar los resultados

    // Cálculo seguro de pasos guardados: ceil(batchSize / stride)
    int savedSteps = (batchSize + stride - 1) / stride;

    // Para Smart: Batch x CellCount (El kernel escribe todo aunque bajemos menos)
    // Para Full: SavedSteps x CellCount (El kernel escribe compactado)
    size_t totalOutputElements;

    if (mode == MODE_FULL_EVOLUTION) {
        totalOutputElements = savedSteps * cellCount;
    } else {
        totalOutputElements = (size_t)batchSize * cellCount;
    }

    resize_buffer_if_needed(&session->d_results, &session->resultCapacityElements, totalOutputElements * 2);

    // 2. Despacho de Estrategia
    if (mode == MODE_FULL_EVOLUTION) {
        run_full_evolution_strategy(session, h_pinned_inflows, h_pinned_results, batchSize, stride, savedSteps);
    } else {
        run_smart_strategy(session, h_pinned_inflows, h_pinned_results, batchSize);
    }

    // SINCRONIZACIÓN FINAL
    // Esperamos a que el stream (Grafos + MemcpyAsync) termine antes de devolver el control a Java.
    // Esto garantiza que los datos en 'h_pinned_results' sean válidos.
    CUDA_CHECK_M(cudaStreamSynchronize(session->stream));
}

void destroy_manning_session(ManningSession* session) {
    if (!session) return;

    if (session->graphExec) cudaGraphExecDestroy(session->graphExec);
    if (session->graph) cudaGraphDestroy(session->graph);
    if (session->stream) cudaStreamDestroy(session->stream);

    if (session->d_bottomWidths) cudaFree(session->d_bottomWidths);
    if (session->d_sideSlopes)   cudaFree(session->d_sideSlopes);
    if (session->d_inv_n)        cudaFree(session->d_inv_n);
    if (session->d_sqrt_slope)   cudaFree(session->d_sqrt_slope);
    if (session->d_pythagoras)   cudaFree(session->d_pythagoras);
    if (session->d_initialQ)      cudaFree(session->d_initialQ);
    if (session->d_initialDepths) cudaFree(session->d_initialDepths);
    if (session->d_results)      cudaFree(session->d_results);
    if (session->d_newInflows)   cudaFree(session->d_newInflows);
    // Liberar Ping-Pong
    if (session->d_ping_Q)       cudaFree(session->d_ping_Q);
    if (session->d_pong_Q)       cudaFree(session->d_pong_Q);
    delete session;
}