package projectstalker.physics.model;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import projectstalker.config.RiverConfig;
import projectstalker.domain.river.RiverGeometry;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.when;

@Slf4j
@ExtendWith(MockitoExtension.class)
class ClimatologicalTemperatureModelTest {

    @Mock
    private RiverGeometry mockGeometry;

    private RiverConfig config;
    private ClimatologicalTemperatureModel model;

    @BeforeEach
    void setUp() {
        // Usamos una configuración base conocida
        config = RiverConfig.getTestingRiver()
                .withAverageAnnualTemperature(15.0f) // T media
                .withSeasonalTempVariation(10.0f)    // +/- 10 grados anual
                .withDailyTempVariation(5.0f);       // +/- 5 grados diario

        // Simulamos un río de 10 celdas
        when(mockGeometry.getCellCount()).thenReturn(10);

        model = new ClimatologicalTemperatureModel(config, mockGeometry);
    }

    @Test
    @DisplayName("Integridad Espacial: El modelo base debe devolver un perfil de temperatura uniforme (plano)")
    void generateProfile_shouldReturnUniformProfile() {
        // ACT
        float[] profile = model.generateProfile(0.0);

        // ASSERT
        // Al ser climatológico puro, todas las celdas del río están a la misma T ambiente
        float firstValue = profile[0];
        for (int i = 1; i < profile.length; i++) {
            assertEquals(firstValue, profile[i], 0.001f,
                    "El perfil base climático debe ser uniforme en todo el río");
        }
    }

    @Test
    @DisplayName("Ciclo Estacional: Verano vs Invierno")
    void generateProfile_shouldReflectSeasonalCycle() {
        // ARRANGE
        // Día 0 (Inicio de año) -> Sin(0) = 0 -> Temp = Media
        // Día 91 (1/4 año, Primavera/Verano aprox en modelo seno) -> Sin(PI/2) = 1 -> Temp = Media + Var
        // Día 273 (3/4 año, Otoño/Invierno) -> Sin(3PI/2) = -1 -> Temp = Media - Var

        double dayInSeconds = 24 * 3600;
        double summerTime = 91.3 * dayInSeconds;
        double winterTime = 273.9 * dayInSeconds;

        // ACT
        float summerTemp = model.generateProfile(summerTime)[0];
        float winterTemp = model.generateProfile(winterTime)[0];

        // ASSERT
        log.info("Temp Verano (Simulada): {}", summerTemp);
        log.info("Temp Invierno (Simulada): {}", winterTemp);

        // Verificamos que verano es más caliente que la media
        assertTrue(summerTemp > 15.0f, "En 'verano' la temperatura debe superar la media");
        // Verificamos que invierno es más frío
        assertTrue(winterTemp < 15.0f, "En 'invierno' la temperatura debe ser inferior a la media");
    }

    // Helper simple
    private void assertTrue(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }
}