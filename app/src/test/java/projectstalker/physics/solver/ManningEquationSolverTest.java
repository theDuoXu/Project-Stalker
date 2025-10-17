package projectstalker.physics.solver;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import projectstalker.domain.river.RiverGeometry;

import java.lang.reflect.Constructor;
import java.lang.reflect.Modifier;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

/**
 * Test unitario para ManningEquationSolver.
 * Se centra en la precisión matemática y la robustez del método Newton-Raphson.
 * Los valores esperados han sido recalculados y corregidos para n=0.030.
 */
class ManningEquationSolverTest {

    private RiverGeometry mockGeometry;

    // --- Parámetros de prueba para un canal de referencia ---
    private static final double TEST_B = 10.0;    // Ancho del fondo (b)
    private static final double TEST_M_RECTANGULAR = 0.0; // Pendiente lateral (m=0)
    private static final double TEST_M_TRAPEZOIDAL = 2.0; // Pendiente lateral (m=2)
    private static final double TEST_N = 0.030;   // Coeficiente de Manning (n)
    private static final double TEST_S = 0.001;   // Pendiente de fondo (S)
    private static final int CELL_INDEX = 0;

    // VALORES ESPERADOS (con n=0.030, S=0.001)
    // El error en los tests originales se debió a que estos valores de referencia eran incorrectos.
    private static final double H_Q5_RECT = 0.6723118;   // Q=5, m=0
    private static final double H_Q100_RECT = 5.1124239; // Q=100, m=0
    private static final double H_Q10_TRAP = 0.9385240;  // Q=10, m=2

    @BeforeEach
    void setUp() {
        mockGeometry = mock(RiverGeometry.class);

        // Configuración BASE: Canal Rectangular (m=0)
        when(mockGeometry.cloneBottomWidth()).thenReturn(new double[]{TEST_B});
        when(mockGeometry.cloneSideSlope()).thenReturn(new double[]{TEST_M_RECTANGULAR}); // m=0
        when(mockGeometry.getManningAt(CELL_INDEX)).thenReturn(TEST_N);
        when(mockGeometry.getBedSlopeAt(CELL_INDEX)).thenReturn(TEST_S);
    }

    // --------------------------------------------------------------------------
    // Test de la estructura de la clase
    // --------------------------------------------------------------------------

    @Test
    @DisplayName("El constructor debe ser privado para prohibir la instanciación")
    void constructorIsPrivate() throws NoSuchMethodException {
        // Se utiliza Reflection para verificar el modificador de acceso.
        Constructor<ManningEquationSolver> constructor = ManningEquationSolver.class.getDeclaredConstructor();
        assertTrue(Modifier.isPrivate(constructor.getModifiers()), "El constructor debe ser privado.");
    }

    // --------------------------------------------------------------------------
    // Test de Escenarios Hidrológicos
    // --------------------------------------------------------------------------

    @Test
    @DisplayName("Convergencia Básica: Caudal medio conocido (Q=5 m³/s)")
    void findDepth_shouldConvergeToKnownValue() {
        // ARRANGE
        final double targetQ = 5.0;
        final double initialGuess = 0.5;

        // ACT
        double calculatedH = ManningEquationSolver.findDepth(targetQ, initialGuess, CELL_INDEX, mockGeometry);

        // ASSERT
        // Se usa la constante H_Q5_RECT
        assertEquals(H_Q5_RECT, calculatedH, 1e-5, "La profundidad calculada debe coincidir con el valor conocido.");
        assertTrue(calculatedH > 0.0, "La profundidad debe ser positiva para un caudal positivo.");
    }

    @Test
    @DisplayName("Caudal Cero/Seco: Debe devolver una profundidad insignificante")
    void findDepth_shouldReturnNearZeroForZeroDischarge() {
        // ARRANGE
        final double targetQ = 1e-7; // Caudal muy cercano a cero
        final double initialGuess = 0.2;

        // ACT
        double calculatedH = ManningEquationSolver.findDepth(targetQ, initialGuess, CELL_INDEX, mockGeometry);

        // ASSERT
        assertTrue(calculatedH < 0.005, "La profundidad para caudal cero debe ser casi cero.");
    }

    @Test
    @DisplayName("Caudal de Crecida: Debe converger rápidamente a una gran profundidad (Q=100 m³/s)")
    void findDepth_shouldConvergeForFloodDischarge() {
        // ARRANGE
        final double targetQ = 100.0;
        final double initialGuess = 1.0;

        // ACT
        double calculatedH = ManningEquationSolver.findDepth(targetQ, initialGuess, CELL_INDEX, mockGeometry);

        // ASSERT
        // Se usa la constante H_Q100_RECT
        assertEquals(H_Q100_RECT, calculatedH, 1e-4, "La profundidad debe ser grande y precisa para el caudal de crecida.");
        assertTrue(calculatedH > initialGuess, "La profundidad debe ser mayor que la estimación inicial.");
    }

    // --------------------------------------------------------------------------
    // Test de Robustez (Edge Cases)
    // --------------------------------------------------------------------------

    @Test
    @DisplayName("Estimación Inicial Negativa/Cero: Debe usar la estimación de 0.1 m")
    void findDepth_shouldHandleNegativeInitialGuess() {
        // ARRANGE
        final double targetQ = 5.0;
        final double negativeGuess = -1.0;

        // ACT
        double calculatedH = ManningEquationSolver.findDepth(targetQ, negativeGuess, CELL_INDEX, mockGeometry);

        // ASSERT
        // Debe converger al valor correcto (H_Q5_RECT) a pesar de la mala estimación.
        assertEquals(H_Q5_RECT, calculatedH, 1e-5, "Debe converger correctamente a pesar de la mala estimación.");
    }

    @Test
    @DisplayName("Pendiente de Fondo Cero: Debe usar el valor saneado de 1e-7")
    void findDepth_shouldHandleZeroBedSlope() {
        // ARRANGE
        final double targetQ = 10.0;
        final double initialGuess = 1.0;

        // Re-mockear la geometría con pendiente cero (activa el saneamiento dentro de findDepth)
        when(mockGeometry.getBedSlopeAt(CELL_INDEX)).thenReturn(0.0);

        // ACT
        double calculatedH = ManningEquationSolver.findDepth(targetQ, initialGuess, CELL_INDEX, mockGeometry);

        // ASSERT: Se verifica que el solver no colapsa y devuelve un valor plausible
        assertTrue(calculatedH > 1.0, "La profundidad debe ser alta o el solver debe converger sin errores.");
        assertFalse(Double.isInfinite(calculatedH) || Double.isNaN(calculatedH), "El solver no debe producir NaN o Infinito.");
    }

    @Test
    @DisplayName("Canal Trapezoidal (m=2.0): Convergencia con geometría compleja")
    void findDepth_shouldHandleTrapezoidalChannel() {
        // ARRANGE
        final double targetQ = 10.0;
        final double initialGuess = 0.5;

        // Re-mockear la geometría para la forma trapezoidal (m=2)
        when(mockGeometry.cloneSideSlope()).thenReturn(new double[]{TEST_M_TRAPEZOIDAL});

        // ACT
        double calculatedH = ManningEquationSolver.findDepth(targetQ, initialGuess, CELL_INDEX, mockGeometry);

        // ASSERT
        // Se usa la constante H_Q10_TRAP
        assertEquals(H_Q10_TRAP, calculatedH, 1e-4, "Debe converger correctamente para el canal trapezoidal.");
    }
}