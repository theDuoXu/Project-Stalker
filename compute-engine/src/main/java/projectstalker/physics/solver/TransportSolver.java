package projectstalker.physics.solver;

import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;

/**
 * Define el contrato para solvers numéricos que resuelven la ecuación de transporte
 * de contaminantes (Advección-Difusión-Reacción) en 1D.
 * 
 * Esquema Numérico Requerido (OE1):
 * Advección: FVM Conservativo + MUSCL 2do Orden + Limitador MinMod.
 * Difusión: Diferencias Centrales (D_L dinámico = alpha * u * H).
 * Reacción: Decaimiento de primer orden.
 * Estabilidad: Sub-stepping adaptativo basado en condición CFL.
 *
 *
 * @author Duo Xu
 * @version 1.0
 * @since 2025-10-14
 */
public interface TransportSolver {

    /**
     * Resuelve la evolución del contaminante para un intervalo de tiempo global {@code dt}.
     * 
     * Responsabilidad de Estabilidad:
     * La implementación NO debe asumir que {@code dt} es seguro. Debe calcular internamente
     * el número de Courant (CFL) y, si es necesario, dividir el paso de tiempo global
     * en múltiples sub-pasos (sub-stepping) para garantizar la estabilidad numérica
     * (CFL menor igual 1.0 o 0.9).
     *
     * @param currentState El estado del río en el tiempo t.
     * Se asume que H (profundidad) y u (velocidad) ya están actualizados
     * para este paso de tiempo o son representativos del intervalo.
     * @param geometry     La geometría estática del río (contiene dx, alpha, k_base).
     * @param dt           El paso de tiempo global a avanzar (segundos).
     * @return Un NUEVO objeto RiverState con la concentración actualizada al tiempo t + dt.
     * Las variables hidráulicas (H, u) se mantienen o copian del estado de entrada.
     */
    RiverState solve(RiverState currentState, RiverGeometry geometry, float dt);

    /**
     * Identificador del solver para logs y benchmarks.
     * @return Nombre técnico (ej: "CPU_MUSCL_Seq", "GPU_SharedMem").
     */
    String getSolverName();
}