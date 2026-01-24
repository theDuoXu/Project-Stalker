package projectstalker.physics.solver.impl;

import lombok.Builder;
import lombok.With;
import lombok.extern.slf4j.Slf4j;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;
import projectstalker.physics.solver.AdvectionSolver;
import projectstalker.physics.solver.DiffusionSolver;
import projectstalker.physics.solver.ReactionSolver;
import projectstalker.physics.solver.TransportSolver;

/**
 * Orquestador principal del transporte de contaminantes en CPU.
 * <p>
 * Responsabilidades:
 * 1. Gestionar la estabilidad numérica (CFL Condition) mediante Sub-stepping.
 * 2. Aplicar la técnica de Operator Splitting (Advección -> Difusión -> Reacción).
 * 3. Coordinar los sub-solvers especializados.
 */
@Slf4j
@With
@Builder
public class SplitOperatorTransportSolver implements TransportSolver {

    private final AdvectionSolver advectionSolver;
    private final DiffusionSolver diffusionSolver;
    private final ReactionSolver reactionSolver;

    // Factor de seguridad CFL (Courant-Friedrichs-Lewy).
    // Para esquemas explícitos, debe ser <= 1.0.
    // Usamos 0.9 para tener un margen de seguridad contra errores de punto flotante.
    // Eliminamos @Builder.Default aquí para inicializarlo explícitamente en el constructor y evitar problemas.
    private double cflSafetyFactor;

    /**
     * Constructor Canónico.
     * Usado por @Builder (si se configura) o manualmente cuando se especifican todos los parámetros.
     *
     * @param advectionSolver Solver de advección.
     * @param diffusionSolver Solver de difusión.
     * @param reactionSolver  Solver de reacción.
     * @param cflSafetyFactor Factor de seguridad CFL (típicamente 0.9). Si es menor igual 0, se fuerza a 0.9.
     */
    public SplitOperatorTransportSolver(AdvectionSolver advectionSolver,
                                        DiffusionSolver diffusionSolver,
                                        ReactionSolver reactionSolver,
                                        double cflSafetyFactor) {
        this.advectionSolver = advectionSolver;
        this.diffusionSolver = diffusionSolver;
        this.reactionSolver = reactionSolver;
        // Validación defensiva: Si entra 0 (por error de builder o inicialización), forzamos 0.9
        this.cflSafetyFactor = (cflSafetyFactor <= 0.0001) ? 0.9 : cflSafetyFactor;
    }

    /**
     * Constructor de conveniencia.
     * Inicializa los solvers dados con el factor CFL por defecto (0.9).
     */
    public SplitOperatorTransportSolver(AdvectionSolver advectionSolver,
                                        DiffusionSolver diffusionSolver,
                                        ReactionSolver reactionSolver) {
        this(advectionSolver, diffusionSolver, reactionSolver, 0.9);
    }

    /**
     * Constructor por defecto.
     * Utiliza las implementaciones estándar (MUSCL, Central, 1er Orden) y CFL 0.9.
     */
    public SplitOperatorTransportSolver() {
        this(new projectstalker.physics.solver.impl.MusclAdvectionSolver(), new projectstalker.physics.solver.impl.CentralDiffusionSolver(), new projectstalker.physics.solver.impl.FirstOrderReactionSolver(), 0.9);
    }

    @Override
    public String getSolverName() {
        return String.format("SplitOp[Adv:%s + Diff:%s + React:%s]",
                advectionSolver.getName(), diffusionSolver.getName(), reactionSolver.getName());
    }

    @Override
    public RiverState solve(RiverState currentState, RiverGeometry geometry, float dt) {
        // 1. Extraer variables hidráulicas (asumimos constantes durante este paso de tiempo)
        float[] u = currentState.velocity();
        float[] h = currentState.waterDepth();
        float[] c = currentState.contaminantConcentration();
        int n = geometry.getCellCount();
        double dx = geometry.getSpatialResolution(); // float -> double implícito

        // Pre-cálculo de áreas mojadas para la advección (se mantiene constante en el sub-stepping)
        float[] areas = new float[n];
        for (int i = 0; i < n; i++) {
            areas[i] = (float) geometry.getCrossSectionalArea(i, h[i]);
        }

        // 2. Determinar Velocidad Máxima para CFL
        float maxVelocity = 0.0f;
        for (float v : u) {
            maxVelocity = Math.max(maxVelocity, Math.abs(v));
        }
        // Evitar división por cero si el agua está estancada
        if (maxVelocity < 1e-5f) maxVelocity = 1e-5f;

        // 3. Cálculo de Sub-stepping (CFL Check)
        double dtMaxAdvection = (dx / (double) maxVelocity) * this.cflSafetyFactor;

        // Protección absoluta contra división por cero o dx inválido
        if (dtMaxAdvection < 1e-4) {
            log.warn("dtMaxAdvection muy bajo ({}). DX={}, Vmax={}, CFL={}. Forzando dtMax=0.1s",
                    dtMaxAdvection, dx, maxVelocity, this.cflSafetyFactor);
            dtMaxAdvection = 0.1;
        }

        // Calculamos número de pasos necesarios para cubrir dt sin violar CFL
        int numSteps = (int) Math.ceil(dt / dtMaxAdvection);
        float dtSub = (float) (dt / numSteps);

        if (numSteps > 1) {
            log.debug("CFL Safety: Sub-stepping activado. {} pasos de {}s (Total: {}s). Vmax={}",
                    numSteps, dtSub, dt, maxVelocity);
        }

        // 4. Bucle de Operator Splitting
        // Trabajamos sobre una copia local de la concentración para no mutar el estado original
        float[] currentC = c.clone();

        for (int step = 0; step < numSteps; step++) {
            // A. Advección
            currentC = advectionSolver.solveAdvection(currentC, u, areas, geometry, dtSub);

            // B. Difusión
            currentC = diffusionSolver.solveDiffusion(currentC, u, h, geometry, dtSub);

            // C. Reacción
            currentC = reactionSolver.solveReaction(currentC, currentState.temperature(), geometry, dtSub);
        }

        // 5. Construir y devolver el nuevo estado final
        return currentState.withContaminantConcentration(currentC);
    }
}