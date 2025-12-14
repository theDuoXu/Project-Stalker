package projectstalker.ui.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpHeaders;
import org.springframework.web.reactive.function.client.ClientRequest;
import org.springframework.web.reactive.function.client.ExchangeFilterFunction;
import org.springframework.web.reactive.function.client.WebClient;
import projectstalker.ui.security.AuthenticationService;

@Configuration
public class ApiClientConfig {

    // Se lee de application.properties
    @Value("${app.api.base-url}")
    private String apiBaseUrl;

    @Bean
    public WebClient apiClient(WebClient.Builder builder, AuthenticationService authService) {
        return builder
                .baseUrl(apiBaseUrl)
                .filter(addBearerToken(authService))
                .build();
    }

    // Filtro que intercepta cada petición y pega el Token si existe
    private ExchangeFilterFunction addBearerToken(AuthenticationService authService) {
        return (request, next) -> {
            String token = authService.getAccessToken();
            if (token != null) {
                ClientRequest authorizedRequest = ClientRequest.from(request)
                        .header(HttpHeaders.AUTHORIZATION, "Bearer " + token)
                        .build();
                return next.exchange(authorizedRequest);
            }
            return next.exchange(request);
        };
    }
}