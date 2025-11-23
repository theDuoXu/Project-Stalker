package projectstalker.physics.impl;

import lombok.Builder;
import lombok.With;
import lombok.extern.slf4j.Slf4j;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;
import projectstalker.physics.i.IAdvectionSolver;
import projectstalker.physics.i.IDiffusionSolver;
import projectstalker.physics.i.IReactionSolver;
import projectstalker.physics.i.ITransportSolver;

/**
 * Orquestador principal del transporte de contaminantes.
 * Responsabilidades:
 * 1. Gestionar la estabilidad numérica (CFL Condition) mediante Sub-stepping.
 * 2. Aplicar la técnica de Operator Splitting (Advección -> Difusión -> Reacción).
 * 3. Coordinar los sub-solvers especializados.
 */
@Slf4j
@With
@Builder
public class SplitOperatorTransportSolver implements ITransportSolver {

    private final IAdvectionSolver advectionSolver;
    private final IDiffusionSolver diffusionSolver;
    private final IReactionSolver reactionSolver;

    // Factor de seguridad CFL (Courant-Friedrichs-Lewy).
    // Para esquemas explícitos, debe ser <= 1.0.
    // Usamos 0.9 para tener un margen de seguridad contra errores de punto flotante.
    @Builder.Default
    private double cflSafetyFactor = 0.9;

    public SplitOperatorTransportSolver(IAdvectionSolver advectionSolver,
                                        IDiffusionSolver diffusionSolver,
                                        IReactionSolver reactionSolver) {
        this.advectionSolver = advectionSolver;
        this.diffusionSolver = diffusionSolver;
        this.reactionSolver = reactionSolver;
    }
    public SplitOperatorTransportSolver(IAdvectionSolver advectionSolver,
                                       IDiffusionSolver diffusionSolver,
                                       IReactionSolver reactionSolver, double cflSafetyFactor) {
        this(advectionSolver, diffusionSolver, reactionSolver);
        this.cflSafetyFactor = cflSafetyFactor;
    }
    public SplitOperatorTransportSolver() {
        this(new MusclAdvectionSolver(), new CentralDiffusionSolver(), new FirstOrderReactionSolver());
    }

    @Override
    public String getSolverName() {
        return String.format("SplitOp[Adv:%s + Diff:%s + React:%s]",
                advectionSolver.getName(), diffusionSolver.getName(), reactionSolver.getName());
    }

    @Override
    public RiverState solve(RiverState currentState, RiverGeometry geometry, float targetDeltaTime) {
        // 1. Extraer variables hidráulicas (asumimos constantes durante este dt)
        float[] u = currentState.velocity();
        float[] h = currentState.waterDepth();
        float[] c = currentState.contaminantConcentration();

        // 2. Calcular variables geométricas necesarias
        int n = geometry.getCellCount();
        double dx = geometry.getSpatial_resolution();

        // Pre-calculamos áreas mojadas para la advección
        float[] areas = new float[n];
        for(int i=0; i<n; i++) {
            areas[i] = (float) geometry.getCrossSectionalArea(i, h[i]);
        }

        // 3. Determinar paso de tiempo seguro (Sub-stepping)
        // Necesitamos encontrar la velocidad máxima para el criterio CFL
        float maxVelocity = 0.0f;
        for (float v : u) {
            maxVelocity = Math.max(maxVelocity, Math.abs(v));
        }

        // Evitar división por cero si el agua está quieta
        if (maxVelocity < 1e-5f) maxVelocity = 1e-5f;

        // Criterio CFL para Advección: dt < dx / u_max
        double dtMaxAdvection = (dx / maxVelocity) * cflSafetyFactor;
        // --- GUARDIA DE SEGURIDAD ---
        // Si dx es 0 o V es infinita, evitamos el bucle de la muerte.
        if (dtMaxAdvection < 1e-5) {
            log.warn("dtMaxAdvection es peligrosamente pequeño ({}). ¿dx=0? Forzando valor mínimo.", dtMaxAdvection);
            dtMaxAdvection = 1e-5;
        }
        // Criterio de Estabilidad para Difusión: dt < dx^2 / (2 * D_max)
        // (Simplificado: solemos dejar que la advección mande porque es más restrictiva en ríos rápidos)

        // Calculamos número de pasos necesarios
        int numSteps = (int) Math.ceil(targetDeltaTime / dtMaxAdvection);
        float dtSub = (float) (targetDeltaTime / numSteps);

        if (numSteps > 1) {
            log.debug("CFL Safety: Sub-stepping activado. {} pasos de {}s (Total: {}s). Vmax={}",
                    numSteps, dtSub, targetDeltaTime, maxVelocity);
        }

        // 4. Bucle de Operator Splitting
        // Trabajamos sobre una copia local de la concentración
        float[] currentC = c.clone();

        for (int step = 0; step < numSteps; step++) {
            // A. Advección
            currentC = advectionSolver.solveAdvection(currentC, u, areas, geometry, dtSub);

            // B. Difusión
            currentC = diffusionSolver.solveDiffusion(currentC, u, h, geometry, dtSub);

            // C. Reacción
            currentC = reactionSolver.solveReaction(currentC, currentState.temperature(), geometry, dtSub);
        }

        // 5. Construir estado final
        // Mantenemos H, u, T, pH inalterados (solo el transporte mueve C)
        return currentState.withContaminantConcentration(currentC);
    }
}