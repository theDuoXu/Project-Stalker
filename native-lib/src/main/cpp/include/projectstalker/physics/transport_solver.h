#pragma once

#include <vector>

/**
 * Orquesta la ejecución de la evolución del transporte (Advección-Difusión-Reacción).
 * Realiza el sub-stepping temporal y gestiona el doble buffer (Ping-Pong) en VRAM.
 *
 * @param h_concentration_in Puntero al array de concentración inicial (Host).
 * @param h_velocity         Puntero al array de velocidades (Host).
 * @param h_depth            Puntero al array de profundidades (Host).
 * @param h_area             Puntero al array de áreas (Host).
 * @param h_temperature      Puntero al array de temperaturas (Host).
 * @param h_alpha            Puntero al array de coeficientes de dispersión (Host).
 * @param h_decay            Puntero al array de coeficientes de decaimiento (Host).
 * @param dx                 Resolución espacial.
 * @param dt_sub             Paso de tiempo seguro para una iteración.
 * @param num_steps          Número de iteraciones a realizar.
 * @param cellCount          Número de celdas.
 * @return Un vector con la concentración final tras la evolución.
 */
std::vector<float> solve_transport_evolution_cpp(
    const float* h_concentration_in,
    const float* h_velocity,
    const float* h_depth,
    const float* h_area,
    const float* h_temperature,
    const float* h_alpha,
    const float* h_decay,
    float dx,
    float dt_sub,
    int num_steps,
    int cellCount
);