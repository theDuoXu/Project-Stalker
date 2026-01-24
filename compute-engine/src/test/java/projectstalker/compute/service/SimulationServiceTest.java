package projectstalker.compute.service;

import org.junit.jupiter.api.Test;
import org.mockito.Mock;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.test.context.ActiveProfiles;
import projectstalker.compute.repository.SimulationRepository;
import projectstalker.config.SimulationConfig;
import projectstalker.domain.simulation.IManningResult;
import projectstalker.domain.simulation.SimulationResponseDTO;
import projectstalker.physics.simulator.ManningSimulator;
import projectstalker.domain.river.RiverGeometry;
import org.springframework.boot.test.autoconfigure.jdbc.AutoConfigureTestDatabase;
import org.springframework.context.annotation.Import;
import projectstalker.compute.TestSecurityConfig;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@SpringBootTest
@ActiveProfiles("mock")
@AutoConfigureTestDatabase(replace = AutoConfigureTestDatabase.Replace.ANY)
@Import(TestSecurityConfig.class)
public class SimulationServiceTest {

    @Autowired
    private SimulationService simulationService;

    @MockBean
    private SimulatorFactory simulatorFactory;

    @MockBean
    private SimulationRepository simulationRepository;

    @MockBean
    private SimulationResultService resultService;

    @Mock
    private ManningSimulator manningSimulator;

    @Test
    void testRunSimulation_Success() {
        // Prepare Mock
        IManningResult mockResult = mock(IManningResult.class);
        when(mockResult.getSimulationTime()).thenReturn(100L);
        when(mockResult.getTimestepCount()).thenReturn(50);

        RiverGeometry geom = mock(RiverGeometry.class);
        when(geom.getCellCount()).thenReturn(100);
        when(mockResult.getGeometry()).thenReturn(geom);

        when(simulatorFactory.createManningSimulator(any())).thenReturn(manningSimulator);
        when(manningSimulator.runFullSimulation()).thenReturn(mockResult);

        // Run
        SimulationConfig config = SimulationConfig.builder()
                .riverConfig(null) // Mocked
                .totalTime(100)
                .deltaTime(1)
                .build();
        SimulationResponseDTO response = simulationService.runSimulation(config);

        // Verify
        assertNotNull(response);
        assertEquals("COMPLETED", response.status());
        verify(simulationRepository, times(2)).save(any()); // Created and Completed
        verify(resultService).saveResult(anyString(), eq(mockResult));
    }
}
