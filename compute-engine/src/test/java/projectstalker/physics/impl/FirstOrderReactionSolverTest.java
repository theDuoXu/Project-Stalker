package projectstalker.physics.impl;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import projectstalker.domain.river.RiverGeometry;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@Slf4j
class FirstOrderReactionSolverTest {

    private FirstOrderReactionSolver solver;
    private RiverGeometry mockGeometry;

    // Constantes para verificación manual
    private static final float THETA = 1.047f;
    private static final float REF_TEMP = 20.0f;
    private static final float K_BASE = 0.5f; // 1/s

    @BeforeEach
    void setUp() {
        // Solver configurado explícitamente
        solver = new FirstOrderReactionSolver(THETA, REF_TEMP, false);

        mockGeometry = mock(RiverGeometry.class);
        when(mockGeometry.getBaseDecayAt(anyInt())).thenReturn(K_BASE);

        log.info("Setup: Solver inicializado. K_BASE={}, Theta={}, RefTemp={}", K_BASE, THETA, REF_TEMP);
    }

    @Test
    @DisplayName("Decaimiento Base: A 20°C debe seguir la fórmula exacta C * exp(-k*dt)")
    void solveReaction_atRefTemp_shouldFollowExactExponentialDecay() {
        log.info(">>> TEST: Decaimiento Isotérmico (20°C)");

        // ARRANGE
        float[] concentration = {100.0f, 50.0f, 0.0f, 10.0f};
        float[] temperature = {20.0f, 20.0f, 20.0f, 20.0f};
        float dt = 1.0f;

        double expectedFactor = Math.exp(-K_BASE * dt); // e^-0.5
        log.info("Factor de decaimiento esperado: {} (para dt={}s)", expectedFactor, dt);

        // ACT
        float[] result = solver.solveReaction(concentration, temperature, mockGeometry, dt);

        // ASSERT
        log.info("Resultados: Entrada -> Salida");
        for (int i = 0; i < concentration.length; i++) {
            float expected = (float) (concentration[i] * expectedFactor);
            log.info("[{}] {} -> {} (Esperado: {})", i, concentration[i], result[i], expected);

            assertEquals(expected, result[i], 1e-6f, "Error en celda " + i);
        }
    }

    @Test
    @DisplayName("Efecto Arrhenius: El calor debe acelerar la reacción")
    void solveReaction_higherTemp_shouldIncreaseDecay() {
        log.info(">>> TEST: Efecto Arrhenius (Calentamiento)");

        float[] concentration = {100.0f};
        float dt = 1.0f;

        // Caso A: 20°C
        float[] resRef = solver.solveReaction(concentration, new float[]{20.0f}, mockGeometry, dt);

        // Caso B: 30°C
        float[] resHot = solver.solveReaction(concentration, new float[]{30.0f}, mockGeometry, dt);

        log.info("C_inicial: 100.0");
        log.info("T=20°C -> C_final: {}", resRef[0]);
        log.info("T=30°C -> C_final: {}", resHot[0]);

        assertTrue(resHot[0] < resRef[0], "El calor debería haber eliminado más contaminante.");

        // Validación Matemática
        double kReal = K_BASE * Math.pow(THETA, 30.0 - 20.0);
        float expectedHot = (float) (100.0f * Math.exp(-kReal * dt));
        log.info("Validación matemática a 30°C: k_real={}, Resultado={}", kReal, expectedHot);

        assertEquals(expectedHot, resHot[0], 1e-5f);
    }

    @Test
    @DisplayName("Efecto Arrhenius: El frío debe frenar la reacción")
    void solveReaction_lowerTemp_shouldDecreaseDecay() {
        log.info(">>> TEST: Efecto Arrhenius (Enfriamiento)");

        float[] concentration = {100.0f};
        float dt = 1.0f;

        float[] resRef = solver.solveReaction(concentration, new float[]{20.0f}, mockGeometry, dt); // 20°C
        float[] resCold = solver.solveReaction(concentration, new float[]{10.0f}, mockGeometry, dt); // 10°C

        log.info("T=20°C -> C_final: {}", resRef[0]);
        log.info("T=10°C -> C_final: {}", resCold[0]);

        assertTrue(resCold[0] > resRef[0], "El frío debería preservar más contaminante.");
    }

    @Test
    @DisplayName("Robustez: Debe manejar arrays de temperatura nulos")
    void solveReaction_nullTemperature_shouldUseReferenceTemp() {
        log.info(">>> TEST: Robustez (Temperatura Null)");

        float[] result = solver.solveReaction(new float[]{100.0f}, null, mockGeometry, 1.0f);

        double expected = 100.0 * Math.exp(-K_BASE * 1.0);
        log.info("Resultado con Temp=null: {} (Debe ser igual a T=20°C: {})", result[0], expected);

        assertEquals((float) expected, result[0], 1e-6f);
    }
}