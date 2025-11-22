#ifndef TRANSPORT_KERNEL_H
#define TRANSPORT_KERNEL_H

#ifdef __cplusplus
extern "C" {
#endif

    /**
     * Lanza el kernel de transporte (Advección + Difusión + Reacción con Arrhenius).
     */
    void launchTransportKernel(
        float* d_c_new,
        const float* d_c_old,
        const float* d_velocity,
        const float* d_depth,
        const float* d_area,
        const float* d_temperature, // <--- NUEVO: Array de Temperatura
        const float* d_alpha,
        const float* d_decay,
        float dx,
        float dt,
        int cellCount
    );

#ifdef __cplusplus
}
#endif

#endif // TRANSPORT_KERNEL_H