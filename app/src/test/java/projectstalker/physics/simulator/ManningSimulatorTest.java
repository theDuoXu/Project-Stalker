package projectstalker.physics.simulator;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Captor;
import org.mockito.Spy;
import org.mockito.junit.jupiter.MockitoExtension;
import projectstalker.config.RiverConfig;
import projectstalker.domain.river.RiverState;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class ManningSimulatorTest {

    private final RiverConfig config = new RiverConfig(
            12345L, 0.0f, 0.05f, 0.001f, 1000.0, 50.0, 200.0, 0.4, 0.0002,
            0.0001, 150.0, 40.0, 4.0, 1.5, 0.030, 0.005, 0.1, 0.05,
            15.0, 2.0, 8.0, 14.0, 7.5, 0.5,
            4.0, 20000.0, 1.5, 1.0, 0.25
    );

    @Spy
    private ManningSimulator simulator = new ManningSimulator(config);

    // --- We will now capture ALL arguments to be thorough ---
    @Captor
    private ArgumentCaptor<float[]> targetDischargesCaptor;
    @Captor
    private ArgumentCaptor<float[]> initialDepthsCaptor;
    @Captor
    private ArgumentCaptor<float[]> bottomWidthsCaptor;
    @Captor
    private ArgumentCaptor<float[]> sideSlopesCaptor;
    @Captor
    private ArgumentCaptor<float[]> manningCoefficientsCaptor;
    @Captor
    private ArgumentCaptor<float[]> bedSlopesCaptor;


    @BeforeEach
    void setUp() {
        // Cualquier llamada a la función será respondida por Mockito
        doReturn(createFakeGpuResult()).when(simulator).solveManningGpu(
                any(float[].class), any(float[].class), any(float[].class),
                any(float[].class), any(float[].class), any(float[].class)
        );
    }

    @Test
    @DisplayName("Should run in CPU mode and transition correctly to GPU mode")
    void shouldRunInCpuModeAndTransitionToGpu() {
        assertThat(simulator.isGpuAccelerated()).isFalse();

        // Run simulation until the river fills up
        while (!simulator.isGpuAccelerated()) {
            simulator.advanceTimeStep(3600.0);
        }

        assertThat(simulator.isGpuAccelerated()).isTrue();
        verify(simulator, never()).solveManningGpu(any(), any(), any(), any(), any(), any());
    }

    @Test
    @DisplayName("Should correctly prepare all data, call GPU method, and reconstruct state")
    void shouldPrepareDataAndReconstructStateOnFirstGpuStep() {
        // 1. Arrange & 2. Act ... (same as before)
        while (!simulator.isGpuAccelerated()) {
            simulator.advanceTimeStep(3600.0);
        }
        simulator.advanceTimeStep(3600.0);

        // 3. Assert: Verify the call AND capture the arguments in one go.
        // This is the correct place to use the captors.
        verify(simulator, times(1)).solveManningGpu(
                targetDischargesCaptor.capture(),
                initialDepthsCaptor.capture(),
                bottomWidthsCaptor.capture(),
                sideSlopesCaptor.capture(),
                manningCoefficientsCaptor.capture(),
                bedSlopesCaptor.capture()
        );

        // 4. Inspect the captured values (this part was already correct)
        int cellCount = simulator.getGeometry().getCellCount();
        float[] capturedDepths = initialDepthsCaptor.getValue();
        assertThat(capturedDepths).hasSize(cellCount);
        for (float depth : capturedDepths) {
            assertThat(depth).isGreaterThanOrEqualTo(0.001f);
        }

        assertThat(capturedDepths).hasSize(cellCount);
        for (float depth : capturedDepths) {
            assertThat(depth).isGreaterThanOrEqualTo(0.001f);
        }

        // Check sanitized bed slopes (CORRECTED ASSERTION)
        float[] capturedSlopes = bedSlopesCaptor.getValue();
        assertThat(capturedSlopes).hasSize(cellCount);
        for (float slope : capturedSlopes) {
            assertThat(slope).isGreaterThanOrEqualTo(1e-7f);
        } // Assert no slope is too small

        // Sanity check on other arrays
        assertThat(targetDischargesCaptor.getValue()).hasSize(cellCount);
        assertThat(bottomWidthsCaptor.getValue()).hasSize(cellCount);


        // 4. Assert: Verify state reconstruction
        RiverState finalState = simulator.getCurrentState();
        assertThat(finalState.getWaterDepthAt(0)).isEqualTo(5.0);
        assertThat(finalState.getVelocityAt(0)).isEqualTo(2.0);
        assertThat(finalState.getTemperatureAt(0)).isNotZero();
    }

    private float[] createFakeGpuResult() {
        int cellCount = simulator.getGeometry().getCellCount();
        float[] fakeResults = new float[cellCount * 2];
        for (int i = 0; i < cellCount; i++) {
            fakeResults[i * 2] = 5.0f;     // Fake depth
            fakeResults[i * 2 + 1] = 2.0f; // Fake velocity
        }
        return fakeResults;
    }
}