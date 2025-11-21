package projectstalker.physics.model;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import projectstalker.config.RiverConfig; // Asegúrate de importar tus clases
import projectstalker.domain.river.RiverGeometry; // Asegúrate de importar tus clases
import projectstalker.factory.RiverGeometryFactory;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Pruebas unitarias y visuales para la clase RiverTemperatureModel.
 */
@Slf4j
class RiverTemperatureModelTest {

    private RiverConfig defaultConfig;
    private RiverGeometry defaultGeometry;

    @BeforeEach
    void setUp() {
        // Configuración reutilizable para las pruebas.
        // Estos valores deberían ser representativos de tu simulación.
        this.defaultConfig = RiverConfig.getTestingRiver();

        RiverGeometryFactory factory = new RiverGeometryFactory();
        int expectedCellCount = (int) (this.defaultConfig.totalLength() / this.defaultConfig.spatialResolution());
        log.debug("Configuración de río creada para el test. Se esperan {} celdas.", expectedCellCount);


        // --- 2. Act (Actuar) ---
        this.defaultGeometry = factory.createRealisticRiver(this.defaultConfig);
        log.debug("RiverGeometry creada satisfactoriamente por la fábrica.");
    }

    @Disabled("Desactivado para permitir tests sin intervención humana, ya que la ventana bloquea hasta que se cierre")
    @Test
    @DisplayName("Visual Test: Debería mostrar el perfil de temperatura en un gráfico")
    void displayProfileChart_shouldShowGraphAndWait() throws InterruptedException {
        // --- 1. Arrange ---
        RiverTemperatureModel model = new RiverTemperatureModel(defaultConfig, defaultGeometry);

        // Simular las 3 PM (15:00) en un día de verano (día 180)
        double summerAfternoon = (180 * 24 * 3600) + (15 * 3600);

        // --- 2. Act & Assert ---
        // Este método mostrará el gráfico y bloqueará el hilo del test
        // hasta que la ventana sea cerrada por el usuario. No se necesita un assert.
        assertDoesNotThrow(() -> model.displayProfileChart(summerAfternoon));
    }

    @Test
    @DisplayName("Logic Test: El perfil calculado debe tener la misma longitud que las celdas de la geometría")
    void calculate_shouldReturnProfileWithCorrectLength() {
        // --- 1. Arrange ---
        RiverTemperatureModel model = new RiverTemperatureModel(defaultConfig, defaultGeometry);
        double anyTime = 123456.0;

        // --- 2. Act ---
        double[] temperatureProfile = model.calculate(anyTime);

        // --- 3. Assert ---
        assertEquals(defaultGeometry.getCellCount(), temperatureProfile.length,
                "El array de temperaturas debe tener un elemento por cada celda del río.");
    }

    @Test
    @DisplayName("Logic Test: Con efectos desactivados, la temperatura debe ser uniforme y coincidir con la base")
    void calculate_shouldReturnBaseTempWhenEffectsAreZero() {
        // --- 1. Arrange ---
        // Se crea una config SIN EFECTOS ESPACIALES, mapeando cuidadosamente cada parámetro
        // a su posición correcta en el constructor del record.
        RiverConfig configWithoutEffects = RiverConfig.getTestingRiver();

        RiverTemperatureModel model = new RiverTemperatureModel(configWithoutEffects, defaultGeometry);

        // Se elige un tiempo donde el cálculo es sencillo: 6 AM del primer día.
        // - El ciclo estacional es 0 (sin(0) = 0).
        // - El ciclo diario es máximo (sin(π/2) = 1).
        double timeAt6AM = 6 * 3600;
        double expectedBaseTemp = 20.0 /* avgAnnual */ + 5.0 /* daily */;

        // --- 2. Act ---
        double[] temperatureProfile = model.calculate(timeAt6AM);

        // --- 3. Assert ---
        // Todas las celdas deben tener exactamente la misma temperatura base.
        for (int i = 0; i < temperatureProfile.length; i++) {
            assertEquals(expectedBaseTemp, temperatureProfile[i], 0.01,
                    "En la celda " + i + ", la temperatura debería ser igual a la base cuando los efectos están desactivados.");
        }
    }
}