#ifndef TRANSPORT_KERNEL_H
#define TRANSPORT_KERNEL_H

#ifdef __cplusplus
extern "C" {
#endif
    // Launcher del Kernel de Baking
    void launchBakingKernel(
        float* d_flow_out,      // u * A
        float* d_diff_out,      // alpha * |u| * h
        float* d_react_out,     // k20 * Arrhenius
        float* d_inv_vol_out,   // 1.0 / (A * dx)
        const float* d_u,
        const float* d_h,
        const float* d_A,
        const float* d_T,
        const float* d_alpha,
        const float* d_decay,
        float dx,
        int cellCount
    );

/**
 * Lanza el kernel de transporte (Advección + Difusión + Reacción con Arrhenius).
 */
void launchTransportKernel(
    float *d_c_new,
    const float *d_c_old,
    const float *d_flow,
    const float *d_diff_coeff,
    const float *d_react_rate,
    const float *d_area,
    float dx,
    float dt,
    int cellCount
);

#ifdef __cplusplus
}
#endif

#endif // TRANSPORT_KERNEL_H
