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

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.anyDouble;
import static org.mockito.Mockito.when;

@Slf4j
@ExtendWith(MockitoExtension.class)
class HeadwaterCoolingDecoratorTest {

    @Mock
    private TemperatureModel mockBaseModel; // El modelo que decoramos
    @Mock
    private RiverGeometry mockGeometry;

    private RiverConfig config;
    private HeadwaterCoolingDecorator decorator;

    @BeforeEach
    void setUp() {
        // Configuración Específica para probar el enfriamiento
        config = RiverConfig.getTestingRiver()
                .withMaxHeadwaterCoolingEffect(5.0f)   // Enfría 5 grados máx
                .withHeadwaterCoolingDistance(100.0f); // El efecto dura 100 metros

        // Simulamos un río de 200 metros (resolución 10m -> 20 celdas)
        when(mockGeometry.getSpatialResolution()).thenReturn(10.0f);

        // Preparamos el mock base: Siempre devuelve 20°C
        // Esto aísla el test: no nos importa si es invierno o verano, solo si la resta funciona.
        float[] baseTemps = new float[20];
        Arrays.fill(baseTemps, 20.0f);
        when(mockBaseModel.generateProfile(anyDouble()))
                .thenReturn(baseTemps);

        decorator = new HeadwaterCoolingDecorator(mockBaseModel, config, mockGeometry);
    }

    @Test
    @DisplayName("Gradiente: El nacimiento (km 0) debe tener el enfriamiento máximo")
    void generateProfile_shouldApplyMaxCoolingAtSource() {
        // ACT
        float[] profile = decorator.generateProfile(0.0);

        // ASSERT
        // Base (20) - MaxCooling (5) = 15
        assertEquals(15.0f, profile[0], 0.001f,
                "En la celda 0, la temperatura debería reducirse en el valor máximo configurado.");
    }

    @Test
    @DisplayName("Gradiente: A mitad de la distancia de enfriamiento, el efecto debe ser del 50%")
    void generateProfile_shouldApplyHalfCoolingAtMidDistance() {
        // ARRANGE
        // Distancia total efecto = 100m. Mitad = 50m.
        // Resolución = 10m. Índice de celda = 5.

        // ACT
        float[] profile = decorator.generateProfile(0.0);

        // ASSERT
        // Factor gradiente = 1.0 - (50m / 100m) = 0.5
        // Enfriamiento = 5.0 * 0.5 = 2.5
        // Resultado = 20 - 2.5 = 17.5
        assertEquals(17.5f, profile[5], 0.001f,
                "A mitad de la distancia de cabecera, el enfriamiento debe ser del 50%.");
    }

    @Test
    @DisplayName("Límite: Más allá de la distancia configurada, no debe haber enfriamiento")
    void generateProfile_shouldStopCoolingAfterDistance() {
        // ARRANGE
        // Distancia efecto = 100m. Probamos a 110m (Índice 11).

        // ACT
        float[] profile = decorator.generateProfile(0.0);

        // ASSERT
        // Debe volver a la temperatura base del mock (20°C)
        assertEquals(20.0f, profile[11], 0.001f,
                "Fuera de la zona de cabecera, la temperatura debe ser la ambiental sin modificaciones.");
    }
}