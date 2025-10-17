package projectstalker.physics.model;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.style.Styler;

import javax.swing.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.util.Arrays;
import java.util.concurrent.CountDownLatch;
import java.util.stream.DoubleStream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Pruebas unitarias y visuales para la clase FlowProfileGenerator.
 */
@Slf4j
class FlowProfileGeneratorTest {
    @Disabled("Desactivado para permitir tests sin intervención humana, ya que la ventana bloquea hasta que se cierre")
    @Test
    @DisplayName("Visual Test: Debería mostrar una gráfica del perfil de caudales")
    void visualizeProfile() throws InterruptedException {
        // --- 1. Configuración ---
        final FlowProfileGenerator generator = new FlowProfileGenerator(
                12345,      // seed
                150.0,      // baseDischarge
                75.0,       // noiseAmplitude
                0.00005f    // noiseFrequency
        );

        // --- 2. Simulación y Visualización ---
        final double totalDays = 10;
        final double endTimeInSeconds = totalDays * 24 * 3600;
        final double timeStepInSeconds = 3600; // Cada hora

        generator.displayProfileChart(0, endTimeInSeconds, timeStepInSeconds);
    }

    @Test
    @DisplayName("Logic Test: Should generate a profile with the correct number of steps")
    void generatesCorrectNumberOfSteps() {
        FlowProfileGenerator generator = new FlowProfileGenerator(1, 100, 50, 0.1f);

        double startTime = 0;
        double endTime = 100;
        double step = 10; // 0, 10, 20, ..., 100 -> 11 pasos

        double[] profile = generator.generateProfile(startTime, endTime, step);

        // El número de pasos es (fin - inicio) / paso + 1
        int expectedSteps = (int) ((endTime - startTime) / step) + 1;
        assertEquals(expectedSteps, profile.length, "The generated profile should have the correct number of data points.");
    }

    @Test
    @DisplayName("Constraint Test: Discharge should never be negative")
    void getDischargeAt_shouldNeverReturnNegative() {
        // Usamos una amplitud muy grande para forzar que el ruido Perlin (-1)
        // lleve el resultado a un valor negativo si no estuviera corregido.
        double baseDischarge = 50.0;
        double largeAmplitude = 100.0;
        FlowProfileGenerator generator = new FlowProfileGenerator(1, baseDischarge, largeAmplitude, 0.1f);

        // Comprobamos en muchos puntos de tiempo
        for (double t = 0; t < 1000; t += 10) {
            double discharge = generator.getDischargeAt(t);
            assertTrue(discharge >= 0, "Discharge at time " + t + " should not be negative, but was " + discharge);
        }
    }

    @Test
    @DisplayName("Logic Test: getTotalVolume should calculate the correct volume for a constant flow")
    void getTotalVolume_shouldCalculateCorrectVolume() {
        // 1. Arrange: Un generador con caudal constante de 10 m³/s
        FlowProfileGenerator generator = new FlowProfileGenerator(1, 10.0, 0.0, 0.1f);

        double startTime = 0;
        double endTime = 100; // 100 segundos
        double timeStep = 1;

        // El volumen esperado es Caudal * Duración = 10 m³/s * 100 s = 1000 m³
        // Hay que tener en cuenta que nuestro método suma N+1 puntos, por lo que el resultado
        // será (N+1) * paso * caudal = 101 * 1 * 10 = 1010.
        double expectedVolume = 10.0 * (endTime - startTime + timeStep);

        // 2. Act: Calcular el volumen usando el método
        double calculatedVolume = generator.getTotalVolume(startTime, endTime, timeStep);

        // 3. Assert: Verificar que el resultado es el esperado
        // Usamos una tolerancia (delta) para evitar problemas con la precisión de los doubles
        assertEquals(expectedVolume, calculatedVolume, 0.001,
                "The calculated total volume should match the expected value for a constant discharge.");
    }

    @Test
    @DisplayName("Logic Test: getPeakDischarge should find the maximum value in the generated profile")
    void getPeakDischarge_shouldFindMaximumValue() {
        // 1. Arrange
        FlowProfileGenerator generator = new FlowProfileGenerator(42, 200, 50, 0.01f);
        double startTime = 0;
        double endTime = 1000;
        double timeStep = 5;

        // 2. Act
        // Primero, generamos el perfil completo para tener una referencia
        double[] profile = generator.generateProfile(startTime, endTime, timeStep);
        // Luego, llamamos al método que queremos probar
        double peakDischarge = generator.getPeakDischarge(startTime, endTime, timeStep);

        // 3. Assert
        // Calculamos el máximo "manualmente" desde el perfil de referencia
        double expectedPeak = java.util.Arrays.stream(profile).max().getAsDouble();

        assertEquals(expectedPeak, peakDischarge, 0.001,
                "getPeakDischarge should return the same maximum value found in the generated profile.");
    }
}