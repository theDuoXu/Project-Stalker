// manning_solver.h
#pragma once

#include <vector>

/**
 * Orquesta la ejecución del kernel de Manning en la GPU.
 * Esta es la función principal que será llamada por la capa JNI.
 *
 * @param initialGuess Vector con las profundidades iniciales.
 * @param flatDischarges Vector con los perfiles de caudal aplanados.
 * @param batchSize El número de pasos de tiempo en el lote.
 * @param cellCount El número de celdas en el río.
 * @param bottomWidths Vector con los anchos de fondo.
 * @param sideSlopes Vector con los taludes.
 * @param manningCoeffs Vector con los coeficientes de Manning.
 * @param bedSlopes Vector con las pendientes del lecho.
 * @return Un vector de floats intercalado con los resultados [D0, V0, D1, V1, ...].
 */
std::vector<float> solve_manning_batch_cpp(
    const std::vector<float>& initialGuess,
    const std::vector<float>& flatDischarges,
    int batchSize,
    int cellCount,
    const std::vector<float>& bottomWidths,
    const std::vector<float>& sideSlopes,
    const std::vector<float>& manningCoeffs,
    const std::vector<float>& bedSlopes
);