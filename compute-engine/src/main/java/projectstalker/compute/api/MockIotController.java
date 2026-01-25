package projectstalker.compute.api;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import projectstalker.config.ApiRoutes;

import java.util.Random;

/**
 * Controller specifically designed to mock IoT devices for the "Real Sensor"
 * webhook demo.
 * Serves plain text numbers compatible with SensorScraper.
 */
@RestController
@RequestMapping("/api/iot/mock-devices")
public class MockIotController {

    private final Random random = new Random();

    @GetMapping("/random")
    public String getRandomValue() {
        // Returns a random number between 10.0 and 30.0
        double val = 10.0 + random.nextDouble() * 20.0;
        return String.format("%.2f", val);
    }

    @GetMapping("/ph")
    public String getPhValue() {
        // Returns valid pH between 6.5 and 8.5
        double val = 6.5 + random.nextDouble() * 2.0;
        return String.format("%.2f", val);
    }

    @GetMapping("/temperature")
    public String getTemperature() {
        // 15 - 25 C
        double val = 15.0 + random.nextDouble() * 10.0;
        return String.format("%.2f", val);
    }

    @GetMapping("/turbidity")
    public String getTurbidity() {
        // 0 - 100 NTU
        double val = random.nextDouble() * 50.0;
        return String.format("%.2f", val);
    }

    @GetMapping("/ammonium")
    public String getAmmonium() {
        // 0.1 - 2.0 mg/L
        double val = 0.1 + random.nextDouble() * 1.9;
        return String.format("%.2f", val);
    }
}
