package projectstalker.compute.repository.mock;

import org.springframework.context.annotation.Primary;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Repository;
import projectstalker.compute.entity.SensorEntity;
import projectstalker.compute.repository.SensorRepository;
import projectstalker.domain.dto.sensor.SensorHealthDTO;
import projectstalker.domain.dto.sensor.SensorReadingDTO;
import projectstalker.domain.sensors.SensorType;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;
import java.util.Random;

@Repository
@Profile("mock")
public class MockSensorRepository implements SensorRepository {

    private final Random random = new Random();

    @Override
    public List<SensorReadingDTO> findReadings(String stationId, String parameter) {
        // Simulación de histórico (la que ya tenías)
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

    @Override
    public List<SensorReadingDTO> findLatestReadings(String stationId) {
        // Simula una "foto" actual de la estación con varios sensores típicos
        // Esto responde al caso "ALL"
        return List.of(
                mockReading(stationId, "TEMPERATURA", 20.0 + random.nextDouble()),
                mockReading(stationId, "PH", 7.0 + random.nextDouble()),
                mockReading(stationId, "OXIGENO_DISUELTO", 8.5 + random.nextDouble()),
                mockReading(stationId, "TURBIDEZ", 5.0 + random.nextDouble())
        );
    }

    @Override
    public List<SensorReadingDTO> findLatestReadingsByType(String stationId, SensorType type) {
        // Simula el dato en tiempo real solo para el tipo solicitado
        double simulatedValue = 10.0 + random.nextDouble(); // Valor arbitrario
        return List.of(
                mockReading(stationId, type.getCode(), simulatedValue)
        );
    }

    @Override
    public List<SensorHealthDTO> findHealthStatus(String stationId) {
        // Simulamos la salud de varios sensores
        return List.of(
                mockHealth(stationId, "TEMPERATURA"),
                mockHealth(stationId, "PH"),
                mockHealth(stationId, "AMONIO"),
                mockHealth(stationId, "TURBIDEZ")
        );
    }

    @Override
    public List<SensorHealthDTO> findHealthStatusByType(String stationId, SensorType type) {
        // Simulamos salud de uno específico
        return List.of(mockHealth(stationId, type.getCode()));
    }

    // --- Helpers privados ---
    private SensorReadingDTO mockReading(String stationId, String paramName, double val) {
        return SensorReadingDTO.builder()
                .tag(stationId + "_" + paramName)
                .timestamp(LocalDateTime.now().toString())
                .value(val)
                .formattedValue(String.format("%.3f", val))
                .stationId(stationId)
                .build();
    }
    // Helper privado para Health
    private SensorHealthDTO mockHealth(String stationId, String paramCode) {
        // Simulamos batería entre 10% y 100% para probar el flag isAllOk a veces false
        int battery = 10 + random.nextInt(91);

        return SensorHealthDTO.builder()
                .tag(stationId + "_" + paramCode)
                .lastChecked(LocalDateTime.now())
                .batteryPercentage(battery)
                .build();
    }

    @Override
    public List<SensorReadingDTO> findReadingsByDateRange(String stationId, String parameter, LocalDateTime from, LocalDateTime to) {
        // Retorna algo simple para probar
        if (from == null) from = LocalDateTime.now().minusDays(1); // Fallback visual

        return List.of(
                mockReading(stationId, parameter, 10.0).withTimestamp(from.toString())
        );
    }

    @Override
    public boolean existsById(String id) {
        return false;
    }

    @Override
    public SensorEntity save(SensorEntity sensor) {
        return null;
    }

    @Override
    public List<SensorEntity> findAllByTwinId(String twinId) {
        return List.of();
    }

    @Override
    public Optional<SensorEntity> findById(String id) {
        return Optional.empty();
    }
}