// manning_kernel.h
#pragma once

// Esto le dice a cualquier compilador de C++ (g++ o nvcc)
// que debe usar enlace C (nombres limpios) para esta función.
#ifdef __cplusplus
extern "C" {
#endif

/**
 * Lanza el kernel de CUDA para resolver la ecuación de Manning para un lote completo.
 * Cada hilo de la GPU resuelve una única celda en un único paso de tiempo.
 *
 * @param d_results Puntero de salida [D0, V0, D1, V1, ...]. Tamaño: totalThreads * 2
 * @param d_initialDepths Puntero de entrada con las estimaciones iniciales. Tamaño: totalThreads
 * @param d_targetDischarges Puntero de entrada con los caudales objetivo. Tamaño: totalThreads
 * @param d_bottomWidths Puntero de entrada a la geometría (Ancho). Tamaño: cellCount
 * @param d_sideSlopes Puntero de entrada a la geometría (Talud). Tamaño: cellCount
 * @param d_manningCoeffs Puntero de entrada a la geometría (Manning). Tamaño: cellCount
 * @param d_bedSlopes Puntero de entrada a la geometría (Pendiente). Tamaño: cellCount
 * @param batchSize Número de pasos de tiempo en el lote.
 * @param cellCount Número de celdas en el río.
 */
void launchManningKernel(
    float* d_results,
    const float* d_initialDepths,
    const float* d_targetDischarges,
    const float* d_bottomWidths,
    const float* d_sideSlopes,
    const float* d_manningCoeffs,
    const float* d_bedSlopes,
    int batchSize,
    int cellCount
);

// --- INICIO DE LA CORRECCIÓN ---
#ifdef __cplusplus
}
#endif
// --- FIN DE LA CORRECCIÓN ---