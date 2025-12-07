package projectstalker.physics.impl;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.mockito.InOrder;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;
import projectstalker.physics.i.IAdvectionSolver;
import projectstalker.physics.i.IDiffusionSolver;
import projectstalker.physics.i.IReactionSolver;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyFloat;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.*;

@Slf4j
class SplitOperatorTransportSolverTest {

    private SplitOperatorTransportSolver orchestrator;

    // Mocks de los especialistas
    private IAdvectionSolver mockAdvection;
    private IDiffusionSolver mockDiffusion;
    private IReactionSolver mockReaction;

    // Datos comunes
    private RiverGeometry mockGeometry;
    private RiverState mockState;
    private final int N = 10;
    private final float DX = 10.0f;

    @BeforeEach
    void setUp() {
        // 1. Crear Mocks
        mockAdvection = mock(IAdvectionSolver.class);
        mockDiffusion = mock(IDiffusionSolver.class);
        mockReaction = mock(IReactionSolver.class);
        mockGeometry = mock(RiverGeometry.class);

        // Configuración del Solver con CFL = 1.0 para facilitar cálculos mentales
        orchestrator = new SplitOperatorTransportSolver(
                mockAdvection, mockDiffusion, mockReaction, 1.0
        );

        // 2. Configurar Geometría
        when(mockGeometry.getCellCount()).thenReturn(N);
        when(mockGeometry.getSpatialResolution()).thenReturn(DX);
        // Mockear cálculo de área para que no devuelva 0
        when(mockGeometry.getCrossSectionalArea(anyInt(), anyDouble())).thenReturn(5.0);

        // 3. Configurar Estado Inicial
        float[] zeros = new float[N];
        // Velocidad 10 m/s para pruebas de CFL
        float[] velocity = new float[N];
        Arrays.fill(velocity, 10.0f);

        // Usamos un RiverState real (no mock) para que funcionen los getters y el .with
        mockState = new RiverState(zeros, velocity, zeros, zeros, zeros);
    }

    @Test
    @DisplayName("Flujo Normal: Sin sub-stepping, debe llamar a la cadena Adv->Diff->React una vez")
    void solve_lowVelocity_shouldExecuteSingleStep() {
        log.info("TEST: Ejecución Simple (1 Paso)");

        // ARRANGE
        // Velocidad = 10 m/s, DX = 10 m.
        // CFL Limit = 1.0 -> dt_max = 10 / 10 = 1.0 segundo.
        // Pedimos dt = 0.5s. (0.5 <= 1.0) -> 1 solo paso.
        float dtRequest = 0.5f;

        // Simulamos que cada solver modifica un poco el array para verificar el encadenamiento
        float[] c_afterAdv = new float[N]; c_afterAdv[0] = 1;
        float[] c_afterDiff = new float[N]; c_afterDiff[0] = 2;
        float[] c_final = new float[N]; c_final[0] = 3;

        when(mockAdvection.solveAdvection(any(), any(), any(), any(), eq(dtRequest))).thenReturn(c_afterAdv);
        when(mockDiffusion.solveDiffusion(same(c_afterAdv), any(), any(), any(), eq(dtRequest))).thenReturn(c_afterDiff);
        when(mockReaction.solveReaction(same(c_afterDiff), any(), any(), eq(dtRequest))).thenReturn(c_final);

        // ACT
        RiverState result = orchestrator.solve(mockState, mockGeometry, dtRequest);

        // ASSERT
        // 1. Verificar resultado final
        assertArrayEquals(c_final, result.contaminantConcentration());

        // 2. Verificar ORDEN de ejecución (Operator Splitting)
        InOrder inOrder = inOrder(mockAdvection, mockDiffusion, mockReaction);
        inOrder.verify(mockAdvection).solveAdvection(any(), any(), any(), any(), eq(dtRequest));
        inOrder.verify(mockDiffusion).solveDiffusion(any(), any(), any(), any(), eq(dtRequest));
        inOrder.verify(mockReaction).solveReaction(any(), any(), any(), eq(dtRequest));
    }

    @Test
    @DisplayName("CFL Safety: Debe activar sub-stepping si la velocidad es alta")
    void solve_highVelocity_shouldTriggerSubStepping() {
        log.info("TEST: Sub-stepping Automático (CFL Check)");

        // ARRANGE
        // Velocidad = 10 m/s, DX = 10 m. dt_max_seguro = 1.0s.
        // Pedimos dt = 2.5s.
        // Pasos necesarios = ceil(2.5 / 1.0) = 3 pasos.
        // dt_real_por_paso = 2.5 / 3 = 0.8333s.
        float dtRequest = 2.5f;
        int expectedSteps = 3;
        float expectedSubDt = dtRequest / expectedSteps;

        // Necesitamos que los mocks devuelvan algo coherente para no romper el bucle
        when(mockAdvection.solveAdvection(any(), any(), any(), any(), anyFloat())).thenAnswer(i -> i.getArgument(0));
        when(mockDiffusion.solveDiffusion(any(), any(), any(), any(), anyFloat())).thenAnswer(i -> i.getArgument(0));
        when(mockReaction.solveReaction(any(), any(), any(), anyFloat())).thenAnswer(i -> i.getArgument(0));

        // ACT
        orchestrator.solve(mockState, mockGeometry, dtRequest);

        // ASSERT
        // Verificamos que cada solver fue llamado 3 veces con el dt dividido
        verify(mockAdvection, times(expectedSteps)).solveAdvection(any(), any(), any(), any(), eq(expectedSubDt));
        verify(mockDiffusion, times(expectedSteps)).solveDiffusion(any(), any(), any(), any(), eq(expectedSubDt));
        verify(mockReaction, times(expectedSteps)).solveReaction(any(), any(), any(), eq(expectedSubDt));

        log.info("Sub-stepping verificado: Se ejecutaron {} pasos de {}s para cubrir {}s totales.", expectedSteps, expectedSubDt, dtRequest);
    }

    @Test
    @DisplayName("Integridad: No debe mutar el estado original")
    void solve_immutabilityCheck() {
        // ARRANGE
        float dt = 0.1f;
        // Mockeamos que la advección cambia los datos
        float[] modifiedConcentration = new float[N];
        Arrays.fill(modifiedConcentration, 999f);
        when(mockAdvection.solveAdvection(any(), any(), any(), any(), anyFloat())).thenReturn(modifiedConcentration);
        when(mockDiffusion.solveDiffusion(any(), any(), any(), any(), anyFloat())).thenReturn(modifiedConcentration);
        when(mockReaction.solveReaction(any(), any(), any(), anyFloat())).thenReturn(modifiedConcentration);

        // ACT
        RiverState result = orchestrator.solve(mockState, mockGeometry, dt);

        // ASSERT
        // 1. El resultado debe tener los datos nuevos
        assertEquals(999f, result.contaminantConcentration()[0]);

        // 2. El estado original (mockState) debe seguir teniendo ceros (inmutable)
        assertEquals(0.0f, mockState.contaminantConcentration()[0], "El estado original fue modificado ilegalmente.");
    }
}