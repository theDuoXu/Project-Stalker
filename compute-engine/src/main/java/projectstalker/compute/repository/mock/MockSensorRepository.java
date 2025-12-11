package projectstalker.compute.repository.mock;

import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Repository;
import projectstalker.compute.repository.SensorRepository;
import projectstalker.domain.sensors.SensorReadingDTO;

import java.util.List;

@Repository
@Profile("mock") // Se activa solo con el perfil 'mock'
public class MockSensorRepository implements SensorRepository {

    @Override
    public List<SensorReadingDTO> findReadings(String stationId, String parameter) {
        // Simulación de que la base de datos/api/scraper o lo que sea nos devuelve 2 filas
        String sensorTag = stationId + "_" + parameter.toUpperCase();

        var r1 = SensorReadingDTO.builder()
                .tag(sensorTag)
                .timestamp("30/10/2025 18:00")
                .value(0.48)
                .formattedValue("0.480")
                .stationId(stationId)
                .build();

        var r2 = SensorReadingDTO.builder()
                .tag(sensorTag)
                .timestamp("30/10/2025 19:00")
                .value(0.52)
                .formattedValue("0.520")
                .stationId(stationId)
                .build();

        return List.of(r1, r2);
    }
}