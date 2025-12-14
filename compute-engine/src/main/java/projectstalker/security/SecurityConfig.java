package projectstalker.security;

import lombok.RequiredArgsConstructor;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpMethod;
import org.springframework.security.config.annotation.method.configuration.EnableMethodSecurity;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configurers.AbstractHttpConfigurer;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.web.SecurityFilterChain;
import projectstalker.config.ApiRoutes;

@Configuration
@EnableWebSecurity
@EnableMethodSecurity(prePostEnabled = true)
@RequiredArgsConstructor
public class SecurityConfig {

    private final KeycloakJwtConverter keycloakJwtConverter;

    // Constantes para Roles
    private static final String ADMIN = "ADMIN";
    private static final String ANALYST = "ANALYST";
    private static final String TECHNICIAN = "TECHNICIAN";
    private static final String OFFICER = "OFFICER";
    private static final String GUEST = "GUEST";

    @Bean
    public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
        http
                .csrf(AbstractHttpConfigurer::disable)
                .sessionManagement(sess -> sess.sessionCreationPolicy(SessionCreationPolicy.STATELESS))

                .authorizeHttpRequests(auth -> auth
                        // 1. PÚBLICO
                        .requestMatchers(
                                ApiRoutes.API_DOCS + "/**",
                                ApiRoutes.SWAGGER_UI + "/**",
                                ApiRoutes.PUBLIC + "/**"
                        ).permitAll()

                        // 2. SENSORES (Concatenamos sub-rutas específicas)
                        .requestMatchers(HttpMethod.GET,
                                ApiRoutes.SENSORS + "/*/realtime",
                                ApiRoutes.SENSORS + "/*/status"
                        ).permitAll()
                        .requestMatchers(HttpMethod.GET,
                                ApiRoutes.SENSORS + "/*/history",
                                ApiRoutes.SENSORS + "/export/**"
                        ).hasAnyRole(GUEST, TECHNICIAN, ANALYST, OFFICER, ADMIN)

                        // 3. ADMIN GLOBAL
                        .requestMatchers(ApiRoutes.ADMIN + "/**").hasRole(ADMIN)

                        // 4. SIMULACIONES & DIGITAL TWINS
                        .requestMatchers(HttpMethod.POST,
                                ApiRoutes.TWINS + "/**",
                                ApiRoutes.SIMULATIONS + "/**"
                        ).hasAnyRole(ANALYST, ADMIN)
                        .requestMatchers(HttpMethod.PUT,
                                ApiRoutes.TWINS + "/**"
                        ).hasAnyRole(ANALYST, ADMIN)
                        .requestMatchers(HttpMethod.DELETE,
                                ApiRoutes.TWINS + "/**"
                        ).hasRole(ADMIN)
                        .requestMatchers(HttpMethod.GET,
                                ApiRoutes.TWINS + "/**",
                                ApiRoutes.SIMULATIONS + "/**"
                        ).hasAnyRole(TECHNICIAN, ANALYST, ADMIN)

                        // 5. OPERACIONES (Alertas e Informes)
                        .requestMatchers(HttpMethod.POST,
                                ApiRoutes.ALERTS + "/manual"
                        ).hasAnyRole(TECHNICIAN, ANALYST, ADMIN)
                        .requestMatchers(HttpMethod.PUT,
                                ApiRoutes.ALERTS + "/**"
                        ).hasAnyRole(OFFICER, ADMIN)
                        .requestMatchers(
                                ApiRoutes.REPORTS + "/**"
                        ).hasAnyRole(OFFICER, ANALYST, ADMIN)

                        // 6. DEFAULT
                        .anyRequest().authenticated()
                )

                // Configuración OAuth2
                .oauth2ResourceServer(oauth2 -> oauth2
                        .jwt(jwt -> jwt.jwtAuthenticationConverter(keycloakJwtConverter))
                );

        return http.build();
    }
}