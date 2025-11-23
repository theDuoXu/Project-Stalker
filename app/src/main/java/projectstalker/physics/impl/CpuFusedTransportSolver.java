package projectstalker.physics.impl;

import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;
import projectstalker.physics.i.IAdvectionSolver;
import projectstalker.physics.i.IDiffusionSolver;
import projectstalker.physics.i.IReactionSolver;
import projectstalker.physics.i.ITransportSolver;

/**
 * Implementación de referencia en CPU que imita el comportamiento del Kernel CUDA "Fused".
 * <p>
 * A diferencia del {@link SplitOperatorTransportSolver} que aplica los pasos secuencialmente
 * (Adv -> Resultado -> Diff), esta clase calcula los cambios de Advección y Difusión
 * basándose en el estado inicial (t) y los suma simultáneamente.
 * <p>
 * La Reacción se aplica secuencialmente al final, igual que en el Kernel v2.
 */
public class CpuFusedTransportSolver implements ITransportSolver {

    private final IAdvectionSolver advectionSolver;
    private final IDiffusionSolver diffusionSolver;
    private final IReactionSolver reactionSolver;

    public CpuFusedTransportSolver() {
        this(new MusclAdvectionSolver(), new CentralDiffusionSolver(), new FirstOrderReactionSolver());
    }

    public CpuFusedTransportSolver(IAdvectionSolver advectionSolver,
                                   IDiffusionSolver diffusionSolver,
                                   IReactionSolver reactionSolver) {
        this.advectionSolver = advectionSolver;
        this.diffusionSolver = diffusionSolver;
        this.reactionSolver = reactionSolver;
    }

    @Override
    public String getSolverName() {
        return "CPU_Fused_Reference";
    }

    @Override
    public RiverState solve(RiverState currentState, RiverGeometry geometry, float dt) {
        float[] u = currentState.velocity();
        float[] h = currentState.waterDepth();
        float[] cOld = currentState.contaminantConcentration();
        int n = geometry.getCellCount();

        // Áreas (necesarias para advección)
        float[] areas = new float[n];
        for(int i=0; i<n; i++) areas[i] = (float) geometry.getCrossSectionalArea(i, h[i]);

        // --- PASO 1: Advección y Difusión SIMULTÁNEAS ---
        // Calculamos qué pasaría si solo hubiera advección
        float[] cAdvOnly = advectionSolver.solveAdvection(cOld, u, areas, geometry, (float) dt);

        // Calculamos qué pasaría si solo hubiera difusión
        float[] cDiffOnly = diffusionSolver.solveDiffusion(cOld, u, h, geometry, (float) dt);

        // Sumamos los "deltas" (Cambios)
        // C_intermedio = C_old + Delta_Adv + Delta_Diff
        float[] cIntermediate = new float[n];
        for (int i = 0; i < n; i++) {
            float deltaAdv = cAdvOnly[i] - cOld[i];
            float deltaDiff = cDiffOnly[i] - cOld[i];

            cIntermediate[i] = cOld[i] + deltaAdv + deltaDiff;

            // Saneamiento intermedio (igual que en GPU)
            if (cIntermediate[i] < 0) cIntermediate[i] = 0.0f;
        }

        // --- PASO 2: Reacción SECUENCIAL ---
        // Aplicamos Arrhenius sobre el resultado transportado (igual que Kernel v2)
        float[] cFinal = reactionSolver.solveReaction(cIntermediate, currentState.temperature(), geometry, (float) dt);

        return currentState.withContaminantConcentration(cFinal);
    }
}