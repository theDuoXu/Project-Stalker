package projectstalker.physics.solver.impl;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.physics.solver.impl.MusclAdvectionSolver.Limiter;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@Slf4j
class MusclAdvectionSolverTest {

    private projectstalker.physics.solver.impl.MusclAdvectionSolver solver;
    private RiverGeometry mockGeometry;

    // Configuración física simple
    private static final float DX = 10.0f; // 10m por celda
    private static final int N = 20;       // 20 celdas

    @BeforeEach
    void setUp() {
        // Usamos MinMod (el más estable) y entrada limpia (0.0)
        solver = new projectstalker.physics.solver.impl.MusclAdvectionSolver(0.0f, Limiter.MINMOD);

        mockGeometry = mock(RiverGeometry.class);
        when(mockGeometry.getSpatialResolution()).thenReturn(DX);
    }

    @Test
    @DisplayName("Transporte Puro: Una onda cuadrada debe moverse a la derecha (u*dt)")
    void solveAdvection_squareWave_shouldMoveRight() {
        log.info(">>> TEST: Transporte de Onda Cuadrada (Advección Pura)");

        // ARRANGE
        float[] concentration = new float[N];
        // Creamos una "caja" de contaminante entre índice 2 y 5
        concentration[2] = 100f; concentration[3] = 100f; concentration[4] = 100f; concentration[5] = 100f;

        // Velocidad constante u = 5 m/s
        float[] velocity = new float[N]; Arrays.fill(velocity, 5.0f);

        // Área constante A = 2 m² (Caudal Q = 10 m³/s)
        float[] area = new float[N]; Arrays.fill(area, 2.0f);

        // Paso de tiempo dt = 2s.
        // Desplazamiento esperado = u * dt = 5 * 2 = 10 metros.
        // Como dx = 10m, la onda debe moverse EXACTAMENTE 1 CELDA a la derecha.
        float dt = 2.0f;

        // ACT
        float[] result = solver.solveAdvection(concentration, velocity, area, mockGeometry, dt);

        // ASSERT & LOGS
        log.info("Inicial: {}", arrayToString(concentration));
        log.info("Final:   {}", arrayToString(result));

        // Verificación de Posición:
        // El pico que estaba en [2] debe estar ahora mayoritariamente en [3]
        // El pico que estaba en [5] debe estar ahora mayoritariamente en [6]

        // Comprobamos que la onda se ha movido
        assertTrue(result[2] < 50f, "La cola de la onda no se ha movido (celda 2 debería vaciarse).");
        assertTrue(result[6] > 50f, "El frente de la onda no ha llegado (celda 6 debería llenarse).");

        // Comprobamos propiedad TVD (No oscilaciones)
        // Ningún valor debe superar el máximo inicial (100) ni ser negativo
        double maxVal = 0;
        double minVal = 0;
        for(float v : result) {
            maxVal = Math.max(maxVal, v);
            minVal = Math.min(minVal, v);
        }

        assertTrue(minVal >= -1e-5, "El solver ha generado valores negativos (Violación TVD).");
        assertTrue(maxVal <= 100.01, "El solver ha generado picos falsos (Violación TVD).");
    }

    @Test
    @DisplayName("CFL Estabilidad: Si dt es demasiado grande (CFL > 1), el solver explota (Responsabilidad del Orquestador)")
    void solveAdvection_highCFL_behavior() {
        // Este test documenta el comportamiento esperado: El solver NO protege contra CFL alto internamente.
        // Eso es trabajo del SplitOperatorTransportSolver. Aquí verificamos que la matemática base hace lo que se le pide.

        float[] c = {0, 100, 0};
        float[] u = {10, 10, 10}; // u=10
        float[] A = {1, 1, 1};
        float dt = 2.0f; // dx=10. u*dt = 20m (2 celdas). CFL = 2.0.

        // Al avanzar 2 celdas en un esquema explícito de 1 celda, esperamos inestabilidad o vaciado excesivo.
        float[] result = solver.solveAdvection(c, u, A, mockGeometry, dt);

        // En un esquema FVM explícito simple, si sale más flujo del que hay (CFL>1), la celda queda negativa.
        log.info("Resultado con CFL=2.0: {}", Arrays.toString(result));

        // Nota: MusclAdvectionSolver tiene un saneamiento "if (new < 0) new = 0",
        // así que no veremos negativos, pero veremos pérdida de masa masiva.
        assertEquals(0.0f, result[1], "Con CFL > 1, la celda donante se vacía completamente (y pierde masa lógica).");
    }

    // Helper para visualización compacta en logs
    private String arrayToString(float[] arr) {
        StringBuilder sb = new StringBuilder("[");
        for (int i=0; i<arr.length; i++) {
            sb.append(String.format("%.0f", arr[i]));
            if (i < arr.length-1) sb.append(", ");
        }
        sb.append("]");
        return sb.toString();
    }
}