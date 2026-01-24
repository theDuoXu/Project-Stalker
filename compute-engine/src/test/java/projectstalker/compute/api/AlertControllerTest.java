package projectstalker.compute.api;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.jdbc.AutoConfigureTestDatabase;
import org.springframework.boot.test.autoconfigure.jdbc.AutoConfigureTestDatabase;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.http.MediaType;
import org.springframework.security.oauth2.jwt.JwtDecoder;
import org.springframework.security.test.context.support.WithMockUser;
import org.springframework.test.web.servlet.MockMvc;
import projectstalker.compute.entity.AlertEntity;
import projectstalker.compute.repository.AlertRepository;

import java.time.LocalDateTime;

import static org.hamcrest.Matchers.hasItem;
import static org.springframework.security.test.web.servlet.request.SecurityMockMvcRequestPostProcessors.jwt;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import projectstalker.compute.TestSecurityConfig;
import org.springframework.context.annotation.Import;

@SpringBootTest
@AutoConfigureMockMvc
@AutoConfigureTestDatabase(replace = AutoConfigureTestDatabase.Replace.ANY)
@Import(TestSecurityConfig.class)
public class AlertControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private AlertRepository alertRepository;

    @MockBean
    private JwtDecoder jwtDecoder;

    @Test
    void testGetAlerts() throws Exception {
        // Setup
        AlertEntity alert = new AlertEntity();
        alert.setSensorId("sensor-1");
        alert.setMessage("Test Alert");
        alert.setSeverity(AlertEntity.AlertSeverity.WARNING);
        alert.setStatus(AlertEntity.AlertStatus.NEW);
        alert.setTimestamp(LocalDateTime.now());
        alertRepository.save(alert);

        mockMvc.perform(get("/alerts")
                .with(jwt()))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$[*].message", hasItem("Test Alert")));
    }

    @Test
    void testAcknowledgeAlert() throws Exception {
        // Setup
        AlertEntity alert = new AlertEntity();
        alert.setSensorId("sensor-1");
        alert.setMessage("To Ack");
        alert.setSeverity(AlertEntity.AlertSeverity.INFO);
        alert.setStatus(AlertEntity.AlertStatus.NEW);
        alert.setTimestamp(LocalDateTime.now());
        alert = alertRepository.save(alert);

        mockMvc.perform(post("/alerts/" + alert.getId() + "/ack")
                .with(jwt()))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status").value("ACKNOWLEDGED"));
    }
}
