#pragma once

#include <vector>

/**
 * Orquesta la ejecución del kernel de Manning en la GPU.
 * Versión optimizada para Zero-Copy (punteros crudos).
 *
 * @param h_initialGuess   Puntero a profundidades iniciales (Host).
 * @param h_flatDischarges Puntero a caudales (Host).
 * @param batchSize        Tamaño del lote.
 * @param cellCount        Número de celdas.
 * @param h_bottomWidths   Puntero directo a geometría (Host/DirectBuffer).
 * @param h_sideSlopes     Puntero directo a geometría.
 * @param h_manningCoeffs  Puntero directo a geometría.
 * @param h_bedSlopes      Puntero directo a geometría.
 */
std::vector<float> solve_manning_batch_cpp(
    const float* h_initialGuess,
    const float* h_flatDischarges,
    int batchSize,
    int cellCount,
    const float* h_bottomWidths,
    const float* h_sideSlopes,
    const float* h_manningCoeffs,
    const float* h_bedSlopes
);