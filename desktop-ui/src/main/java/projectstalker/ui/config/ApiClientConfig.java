package projectstalker.ui.config;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpHeaders;
import org.springframework.web.reactive.function.client.ClientRequest;
import org.springframework.web.reactive.function.client.ExchangeFilterFunction;
import org.springframework.web.reactive.function.client.WebClient;
import projectstalker.ui.security.AuthenticationService;

@Configuration
@Slf4j
public class ApiClientConfig {

    // Se lee de application.properties
    @Value("${app.api.base-url}")
    private String apiBaseUrl;

    @Bean
    public WebClient apiClient(WebClient.Builder builder, AuthenticationService authService) {
        reactor.netty.http.client.HttpClient httpClient = reactor.netty.http.client.HttpClient.create()
                .responseTimeout(java.time.Duration.ofSeconds(60));

        return builder
                .baseUrl(apiBaseUrl)
                .clientConnector(new org.springframework.http.client.reactive.ReactorClientHttpConnector(httpClient))
                .filter(addBearerToken(authService))
                .build();
    }

    // Filtro que intercepta cada petición y pega el Token si existe
    private ExchangeFilterFunction addBearerToken(AuthenticationService authService) {
        return (request, next) -> {
            String token = authService.getAccessToken();

            // --- LOG DE DEPURACIÓN ---
            if (token != null && !token.isBlank()) {
                log.info("AUTH: Adjuntando Bearer Token. (Empieza por: {}...)", token.charAt(0));

                ClientRequest authorizedRequest = ClientRequest.from(request)
                        .header(HttpHeaders.AUTHORIZATION, "Bearer " + token)
                        .build();
                return next.exchange(authorizedRequest);
            } else {
                log.error("AUTH: ¡Token NULL o Vacío! Enviando petición anónima (Esto fallará con 401).");
                return next.exchange(request);
            }
        };
    }
}