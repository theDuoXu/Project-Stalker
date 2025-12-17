package projectstalker.compute.integration;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.security.oauth2.jwt.JwtDecoder;
import org.springframework.test.context.ActiveProfiles;
import projectstalker.compute.service.DigitalTwinService;
import projectstalker.config.RiverConfig;
import projectstalker.domain.dto.twin.TwinCreateRequest;
import projectstalker.domain.dto.twin.TwinDetailDTO;
import projectstalker.domain.dto.twin.TwinSummaryDTO;
import projectstalker.domain.event.GeoEvManMadeDam;
import projectstalker.domain.event.GeologicalEvent;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

@Tag("Integration")
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.NONE)
@ActiveProfiles("mock")
class DigitalTwinIntegrationTest {

    @Autowired
    private DigitalTwinService twinService;

    // 2. SOLUCIÓN AL ERROR JwtDecoder
    // Spring Security necesita este bean para arrancar.
    // Al poner @MockBean, Spring crea uno falso y vacío.
    // Como en este test no llamamos a la API (MockMvc), sino al Servicio directo,
    // este decoder nunca se usa, pero permite que la app arranque.
    @MockBean
    private JwtDecoder jwtDecoder;

    @Test
    void shouldCreateAndPersistDigitalTwinWithJsonbAndPolymorphism() {
        // 1. GIVEN
        var config = RiverConfig.getTestingRiver();

        // Creamos una presa (Evento con atributos específicos)
        var dam = new GeoEvManMadeDam(500.0f, 215.0f, 200.0f, 5);
        List<GeologicalEvent> events = List.of(dam);

        var request = new TwinCreateRequest(
                "Rio Integration Test",
                "Verificando persistencia JSONB y herencia de Jackson",
                config,
                events
        );

        // 2. WHEN - Guardar
        TwinSummaryDTO created = twinService.createTwin(request);

        // 3. THEN - Verificar creación básica
        assertThat(created.id()).isNotNull();
        assertThat(created.name()).isEqualTo("Rio Integration Test");
        assertThat(created.cellCount()).isGreaterThan(100);

        // 4. WHEN - Leer (Recuperar JSONB)
        TwinDetailDTO details = twinService.getTwinDetails(created.id());

        // 5. THEN - Verificar datos complejos
        assertThat(details.config()).isNotNull();
        assertThat(details.config().baseWidth()).isEqualTo(150.0f);

        // Verificar Polimorfismo
        assertThat(details.events()).hasSize(1);
        assertThat(details.events().getFirst()).isInstanceOf(GeoEvManMadeDam.class);

        GeoEvManMadeDam savedDam = (GeoEvManMadeDam) details.events().getFirst();
        assertThat(savedDam.getReservoirWidth()).isEqualTo(200.0f);
    }
}