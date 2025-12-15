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

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.anyDouble;
import static org.mockito.Mockito.when;

@Slf4j
@ExtendWith(MockitoExtension.class)
class RiverPhModelTest {

    @Mock
    private RiverGeometry mockGeometry;

    @Mock
    private TemperatureModel mockTempModel;

    private RiverPhModel phModel;
    private RiverConfig config;

    // Datos base para el mock
    private final float[] BASE_GEO_PH = new float[]{7.0f, 7.0f, 7.0f, 7.0f, 7.0f}; // pH neutro geológico
    private final float[] BASE_DECAY = new float[]{0.1f, 0.1f, 0.1f, 0.1f, 0.1f}; // Actividad biológica estándar

    @BeforeEach
    void setUp() {
        // 1. Configuración del Río (Usamos el helper de test)
        config = RiverConfig.getTestingRiver();
        // config.baseDecayRateAt20C() es 0.1 en el TestingRiver

        // 2. Mock de Geometría (Datos Estáticos)
        when(mockGeometry.getCellCount()).thenReturn(BASE_GEO_PH.length);
        when(mockGeometry.getPhProfile()).thenReturn(BASE_GEO_PH); // Devuelve la referencia al array base
        when(mockGeometry.getBaseDecayCoefficientAt20C()).thenReturn(BASE_DECAY);

        // 3. Inicializar el modelo con los mocks
        phModel = new RiverPhModel(config, mockGeometry, mockTempModel);
    }

    @Test
    @DisplayName("Ciclo Diario: El pH debe ser mayor por la tarde (fotosíntesis) que por la mañana")
    void generateProfile_shouldReflectDailyCycle() {
        // ARRANGE
        // Fijamos una temperatura constante de 20°C para aislar el efecto del ciclo solar
        float[] constantTemp = new float[]{20f, 20f, 20f, 20f, 20f};
        when(mockTempModel.generateProfile(anyDouble())).thenReturn(constantTemp);

        // Tiempos clave (Basado en PHASE_SHIFT_HOURS = 15.0)
        double peakTime = 15.0 * 3600.0; // 15:00 PM (Pico esperado)
        double lowTime = 3.0 * 3600.0;   // 03:00 AM (Valle esperado)

        // ACT
        float[] phAtPeak = phModel.generateProfile(peakTime);
        float[] phAtLow = phModel.generateProfile(lowTime);

        // ASSERT
        log.info("pH a las 15:00 (Pico): {}", phAtPeak[0]);
        log.info("pH a las 03:00 (Valle): {}", phAtLow[0]);

        // Verificamos física básica: Fotosíntesis consume CO2 -> Sube pH
        assertTrue(phAtPeak[0] > 7.0f, "A las 15:00 el pH debería ser mayor que la base (7.0)");
        assertTrue(phAtLow[0] < 7.0f, "A las 03:00 el pH debería ser menor que la base (7.0)");
        assertTrue(phAtPeak[0] > phAtLow[0], "El pH de la tarde debe ser mayor que el de la madrugada");
    }

    @Test
    @DisplayName("Efecto Térmico: Mayor temperatura debe aumentar la amplitud de oscilación del pH")
    void generateProfile_shouldAmplifyWithTemperature() {
        // ARRANGE
        double peakTime = 15.0 * 3600.0; // Usamos el momento de mayor efecto

        // Caso 1: Agua Fría (10°C) -> Metabolismo lento
        when(mockTempModel.generateProfile(anyDouble())).thenReturn(new float[]{10f, 10f, 10f, 10f, 10f});
        float[] phCold = phModel.generateProfile(peakTime);

        // Caso 2: Agua Caliente (30°C) -> Metabolismo rápido (Arrhenius)
        when(mockTempModel.generateProfile(anyDouble())).thenReturn(new float[]{30f, 30f, 30f, 30f, 30f});
        float[] phHot = phModel.generateProfile(peakTime);

        // ASSERT
        float deltaCold = phCold[0] - 7.0f; // Cuánto subió respecto a la base
        float deltaHot = phHot[0] - 7.0f;

        log.info("Incremento de pH a 10°C: {}", deltaCold);
        log.info("Incremento de pH a 30°C: {}", deltaHot);

        assertTrue(deltaHot > deltaCold,
                "El pH debería subir más en agua caliente debido a la mayor actividad biológica");
    }

    @Test
    @DisplayName("Integridad de Datos: El array devuelto es nuevo y no afecta a la geometría base")
    void generateProfile_shouldReturnNewArrayInstance() {
        // ARRANGE
        when(mockTempModel.generateProfile(anyDouble())).thenReturn(new float[]{20f, 20f, 20f, 20f, 20f});

        // ACT
        float[] resultProfile = phModel.generateProfile(0.0);

        // Mutamos el resultado
        resultProfile[0] = 14.0f;

        // ASSERT
        // Verificamos que el array mockeado original NO ha cambiado
        assertEquals(7.0f, BASE_GEO_PH[0],
                "El perfil base en la geometría no debe verse afectado por cambios en el resultado de la simulación");

        // Verificamos que una segunda llamada devuelve valores calculados frescos
        float[] secondResult = phModel.generateProfile(0.0);
        assertNotEquals(14.0f, secondResult[0],
                "Cada llamada debe generar una nueva instancia de datos");
    }
}