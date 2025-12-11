package projectstalker.physics.solver.impl;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import projectstalker.domain.river.RiverGeometry;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@Slf4j
class CentralDiffusionSolverTest {

    private projectstalker.physics.solver.impl.CentralDiffusionSolver solver;
    private RiverGeometry mockGeometry;

    // Parámetros de prueba
    private static final float DX = 10.0f; // 10 metros por celda
    private static final float ALPHA = 1.0f; // Coeficiente de Taylor simple
    private static final float MIN_DIFFUSION = 1e-6f;

    @BeforeEach
    void setUp() {
        solver = new projectstalker.physics.solver.impl.CentralDiffusionSolver(MIN_DIFFUSION);

        mockGeometry = mock(RiverGeometry.class);
        when(mockGeometry.getSpatialResolution()).thenReturn(DX);
        // Mockeamos alpha constante para simplificar cálculos manuales
        when(mockGeometry.getDispersionAlphaAt(anyInt())).thenReturn(ALPHA);
    }

    @Test
    @DisplayName("Difusión Estándar: Un pico central debe ensancharse simétricamente")
    void solveDiffusion_peakConcentration_shouldSpreadSymmetrically() {
        log.info("--- TEST: Ensanchamiento Gaussiano (La gota de tinta)");

        // ARRANGE
        // 5 Celdas: [0, 0, 100, 0, 0] -> Pico en el centro (i=2)
        float[] concentration = {0.0f, 0.0f, 100.0f, 0.0f, 0.0f};

        // Condiciones hidráulicas constantes
        float[] velocity = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f}; // 1 m/s
        float[] depth = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f};    // 2 m de profundidad

        float dt = 1.0f; // 1 segundo

        // CÁLCULO MANUAL ESPERADO:
        // 1. D_L = alpha * u * h = 1.0 * 1.0 * 2.0 = 2.0 m²/s
        // 2. r (Número de difusión) = (D_L * dt) / dx² = (2.0 * 1.0) / 100.0 = 0.02
        // 3. Esquema: C_new = C_old + r * (C_next - 2*C_curr + C_prev)

        // Para celda central (i=2): 100 + 0.02 * (0 - 200 + 0) = 100 - 4 = 96.0
        // Para vecinos (i=1 y i=3): 0 + 0.02 * (100 - 0 + 0) = 2.0

        // ACT
        float[] result = solver.solveDiffusion(concentration, velocity, depth, mockGeometry, dt);

        // ASSERT & LOGS
        log.info("Estado Inicial: {}", Arrays.toString(concentration));
        log.info("Estado Final:   {}", Arrays.toString(result));

        // 1. El pico debe bajar
        assertEquals(96.0f, result[2], 1e-5f, "El pico central no bajó lo esperado.");

        // 2. Los vecinos deben subir (simetría)
        assertEquals(2.0f, result[1], 1e-5f, "El vecino izquierdo no recibió masa.");
        assertEquals(2.0f, result[3], 1e-5f, "El vecino derecho no recibió masa.");

        // 3. Simetría perfecta
        assertEquals(result[1], result[3], "La difusión debe ser simétrica si la velocidad es constante.");
    }

    @Test
    @DisplayName("Conservación de Masa: La cantidad total de contaminante debe mantenerse (Sistema Aislado)")
    void solveDiffusion_shouldConserveMass() {
        log.info(">>> TEST: Conservación de Masa");

        // ARRANGE
        // Ceros extra en los extremos para que la difusión
        // no llegue a tocar la condición de frontera en este paso de tiempo.
        float[] concentration = {0.0f, 0.0f, 0.0f, 50.0f, 20.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        // La difusión de 50.0 afectará a los índices 2 y 4.
        // El índice 1 (vecino del 2) depende de (0,0,50), así que cambiará?
        // En t=0, C[1]=0. Sus vecinos son C[0]=0 y C[2]=0.
        // POR TANTO: C[1] se mantendrá en 0 en el primer paso. La frontera C[0] copiará 0.

        float[] velocity = new float[9]; Arrays.fill(velocity, 1.0f);
        float[] depth = new float[9];    Arrays.fill(depth, 1.0f);
        float dt = 1.0f; // Reducimos un poco dt para asegurar que no difunda hasta el borde

        // ACT
        float[] result = solver.solveDiffusion(concentration, velocity, depth, mockGeometry, dt);

        // ASSERT
        double initialMass = sumArray(concentration);
        double finalMass = sumArray(result);

        log.info("Masa Inicial: {}", initialMass);
        log.info("Masa Final:   {}", finalMass);
        log.info("Estado Final: {}", Arrays.toString(result));

        // Verificamos que los bordes siguen siendo cero (la mancha no tocó la pared)
        assertEquals(0.0f, result[0], 1e-6f, "La mancha tocó el borde izquierdo, invalidando el test de masa.");
        assertEquals(0.0f, result[result.length-1], 1e-6f, "La mancha tocó el borde derecho.");

        assertEquals(initialMass, finalMass, 1e-4f, "La difusión ha creado o destruido masa ilegalmente.");
    }

    @Test
    @DisplayName("Suelo de Difusión: Si el agua está quieta, debe aplicarse la difusión mínima")
    void solveDiffusion_zeroVelocity_shouldUseMinDiffusion() {
        log.info("--- TEST: Agua Estancada (Velocidad Cero)");

        // ARRANGE
        float[] concentration = {0.0f, 0.0f, 100.0f, 0.0f, 0.0f};
        float[] velocity = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; // VELOCIDAD CERO -> D_L teórico = 0
        float[] depth = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
        float dt = 1000.0f; // Mucho tiempo para que se note la difusión mínima

        // ACT
        float[] result = solver.solveDiffusion(concentration, velocity, depth, mockGeometry, dt);

        // ASSERT
        log.info("Pico inicial: 100.0 -> Pico final: {}", result[2]);

        assertTrue(result[2] < 100.0f, "El pico debería haber bajado por difusión molecular mínima.");
        assertTrue(result[1] > 0.0f, "Los vecinos deberían haber recibido algo de masa.");

        // Verificamos que no explotó (NaN)
        assertFalse(Float.isNaN(result[2]));
    }

    @Test
    @DisplayName("Condiciones de Frontera: Los bordes deben copiar a sus vecinos (Neumann)")
    void solveDiffusion_boundaryConditions() {
        log.info("--- TEST: Fronteras (Neumann)");

        // ARRANGE
        float[] concentration = {10.0f, 20.0f, 20.0f, 20.0f, 50.0f};
        //                       ^ borde                 ^ borde
        float[] velocity = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
        float[] depth = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
        float dt = 0.1f;

        // ACT
        float[] result = solver.solveDiffusion(concentration, velocity, depth, mockGeometry, dt);

        // ASSERT
        // El solver implementa result[0] = result[1] y result[N-1] = result[N-2]
        log.info("Borde Izquierdo: [0]={}, [1]={}", result[0], result[1]);
        log.info("Borde Derecho:   [N-1]={}, [N-2]={}", result[4], result[3]);

        assertEquals(result[1], result[0], 1e-6f, "Frontera izquierda debe ser Neumann (copia).");
        assertEquals(result[3], result[4], 1e-6f, "Frontera derecha debe ser Neumann (copia).");
    }

    // Helper para sumar masa
    private double sumArray(float[] arr) {
        double sum = 0;
        for (float v : arr) sum += v;
        return sum;
    }
}