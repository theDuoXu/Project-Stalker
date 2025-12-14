package projectstalker.compute.api;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.autoconfigure.data.jpa.JpaRepositoriesAutoConfiguration;
import org.springframework.boot.autoconfigure.jdbc.DataSourceAutoConfiguration;
import org.springframework.boot.autoconfigure.orm.jpa.HibernateJpaAutoConfiguration;
import org.springframework.boot.autoconfigure.security.oauth2.resource.servlet.OAuth2ResourceServerAutoConfiguration;
import org.springframework.boot.autoconfigure.security.servlet.SecurityAutoConfiguration;
import org.springframework.boot.autoconfigure.security.servlet.SecurityFilterAutoConfiguration;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.context.annotation.FilterType;
import org.springframework.http.MediaType;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.web.servlet.MockMvc;
import projectstalker.compute.service.SensorService;
import projectstalker.config.ApiRoutes;
import projectstalker.security.SecurityConfig;
import projectstalker.domain.dto.sensor.SensorHealthDTO;
import projectstalker.domain.dto.sensor.SensorHealthResponseDTO;
import projectstalker.domain.dto.sensor.SensorReadingDTO;
import projectstalker.domain.dto.sensor.SensorResponseDTO;

import java.time.LocalDateTime;
import java.util.List;

import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.BDDMockito.given;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultHandlers.print;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@WebMvcTest(
        controllers = SensorController.class,
        excludeFilters = @ComponentScan.Filter(type = FilterType.ASSIGNABLE_TYPE, classes = SecurityConfig.class),
        excludeAutoConfiguration = {
                // --- Seguridad ---
                SecurityAutoConfiguration.class,
                SecurityFilterAutoConfiguration.class,
                OAuth2ResourceServerAutoConfiguration.class,
                // --- Base de Datos y JPA ---
                DataSourceAutoConfiguration.class,
                JpaRepositoriesAutoConfiguration.class,
                HibernateJpaAutoConfiguration.class
        }
)
@AutoConfigureMockMvc(addFilters = false)
@ActiveProfiles("mock")
class SensorControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockBean
    private SensorService sensorService;

    // --- TEST 1: HISTÓRICO ---
    @Test
    void getSensorHistory_ShouldReturn200_WhenParamsAreValid() throws Exception {
        // A. GIVEN
        String stationId = "C302";
        String param = "AMONIO";

        var reading = SensorReadingDTO.builder()
                .tag("C302_AMONIO")
                .timestamp("30/10/2025 18:00")
                .value(0.48)
                .formattedValue("0.480")
                .stationId(stationId)
                .build();

        var responseMock = SensorResponseDTO.builder()
                .name("AMONIO")
                .unit("mg/l")
                .signalType("ISE")
                .values(List.of(reading))
                .build();

        given(sensorService.getHistory(stationId, param)).willReturn(responseMock);

        // B. WHEN & THEN
        mockMvc.perform(get(ApiRoutes.SENSORS + "/{stationId}/history", stationId)
                        .param("parameter", param)
                        .contentType(MediaType.APPLICATION_JSON))
                .andDo(print())
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.name").value("AMONIO"))
                .andExpect(jsonPath("$.unit").value("mg/l"))
                .andExpect(jsonPath("$.values[0].value").value(0.48));
    }

    // --- TEST 2: REALTIME (Lista) ---
    @Test
    void getStationRealtime_ShouldReturnList_WhenCalled() throws Exception {
        // A. GIVEN
        String stationId = "C302";
        String param = "ALL";

        var readingPh = SensorReadingDTO.builder()
                .tag("C302_PH")
                .timestamp("30/10/2025 18:05")
                .value(7.4)
                .formattedValue("7.40")
                .stationId(stationId)
                .build();

        given(sensorService.getRealtime(stationId, param)).willReturn(List.of(readingPh));

        // B. WHEN & THEN
        mockMvc.perform(get(ApiRoutes.SENSORS + "/{stationId}/realtime", stationId)
                        .param("parameter", param)
                        .contentType(MediaType.APPLICATION_JSON))
                .andDo(print())
                .andExpect(status().isOk())
                .andExpect(jsonPath("$").isArray())
                .andExpect(jsonPath("$[0].tag").value("C302_PH"))
                .andExpect(jsonPath("$[0].value").value(7.4));
    }

    // --- TEST 3: STATUS / HEALTH ---
    @Test
    void getStationStatus_ShouldReturnHealthResponse_WhenCalled() throws Exception {
        // A. GIVEN
        String stationId = "C302";
        String param = "ALL";

        var healthItem = SensorHealthDTO.builder()
                .tag("C302_PH")
                .lastChecked(LocalDateTime.now())
                .batteryPercentage(85)
                .build();

        var healthResponse = SensorHealthResponseDTO.builder()
                .stationId(stationId)
                .isAllOk(true)
                .values(List.of(healthItem))
                .build();

        given(sensorService.getHealth(stationId, param)).willReturn(healthResponse);

        // B. WHEN & THEN
        mockMvc.perform(get(ApiRoutes.SENSORS + "/{stationId}/status", stationId)
                        .param("parameter", param)
                        .contentType(MediaType.APPLICATION_JSON))
                .andDo(print())
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.stationId").value(stationId))
                .andExpect(jsonPath("$.isAllOk").value(true));
//                .andExpect(jsonPath("$.values[0].batteryPercentage").value(85));
    }

    // --- TEST 4: EXPORT UNITARIO ---
    @Test
    void exportReadings_ShouldReturnData_WhenDatesAreValid() throws Exception {
        // A. GIVEN
        String stationId = "C302";
        String param = "AMONIO";
        // Definimos las fechas como Strings ISO-8601 (lo que enviará el frontend)
        String fromStr = "2025-10-01T00:00:00";
        String toStr = "2025-10-02T00:00:00";

        // Lo que esperamos que reciba el servicio (convertido a objeto Java)
        LocalDateTime from = LocalDateTime.parse(fromStr);
        LocalDateTime to = LocalDateTime.parse(toStr);

        var mockExport = SensorResponseDTO.builder()
                .name("AMONIO")
                .unit("mg/l")
                .values(List.of()) // Lista vacía para simplificar
                .build();

        // Verificamos que el Controller convierta bien los strings a LocalDateTime antes de llamar al servicio
        given(sensorService.getExportData(eq(stationId), eq(param), eq(from), eq(to)))
                .willReturn(mockExport);

        // B. WHEN & THEN
        mockMvc.perform(get(ApiRoutes.SENSORS + "/export/{stationId}", stationId)
                        .param("parameter", param)
                        .param("from", fromStr)
                        .param("to", toStr)
                        .contentType(MediaType.APPLICATION_JSON))
                .andDo(print())
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.name").value("AMONIO"))
                .andExpect(jsonPath("$.unit").value("mg/l"));
    }
}