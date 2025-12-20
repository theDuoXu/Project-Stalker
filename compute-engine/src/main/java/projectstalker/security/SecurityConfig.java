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
                        ).hasAnyRole(
                                UserRole.GUEST.name(),
                                UserRole.TECHNICIAN.name(),
                                UserRole.ANALYST.name(),
                                UserRole.OFFICER.name(),
                                UserRole.ADMIN.name()
                        )

                        // 3. ADMIN GLOBAL
                        .requestMatchers(ApiRoutes.ADMIN + "/**").hasRole(UserRole.ADMIN.name())

                        // 4. SIMULACIONES & DIGITAL TWINS
                        .requestMatchers(HttpMethod.POST,
                                ApiRoutes.TWINS + "/**",
                                ApiRoutes.SIMULATIONS + "/**"
                        ).hasAnyRole(UserRole.ANALYST.name(), UserRole.ADMIN.name())
                        .requestMatchers(HttpMethod.PUT,
                                ApiRoutes.TWINS + "/**"
                        ).hasAnyRole(UserRole.ANALYST.name(), UserRole.ADMIN.name())
                        .requestMatchers(HttpMethod.DELETE,
                                ApiRoutes.TWINS + "/**"
                        ).hasRole(UserRole.ADMIN.name())
                        .requestMatchers(HttpMethod.GET,
                                ApiRoutes.TWINS + "/**",
                                ApiRoutes.SIMULATIONS + "/**"
                        ).hasAnyRole(UserRole.TECHNICIAN.name(), UserRole.ANALYST.name(), UserRole.ADMIN.name())

                        // 5. OPERACIONES (Alertas e Informes)
                        .requestMatchers(HttpMethod.POST,
                                ApiRoutes.ALERTS + "/manual"
                        ).hasAnyRole(UserRole.TECHNICIAN.name(), UserRole.ANALYST.name(), UserRole.ADMIN.name())
                        .requestMatchers(HttpMethod.PUT,
                                ApiRoutes.ALERTS + "/**"
                        ).hasAnyRole(UserRole.OFFICER.name(), UserRole.ADMIN.name())
                        .requestMatchers(
                                ApiRoutes.REPORTS + "/**"
                        ).hasAnyRole(UserRole.OFFICER.name(), UserRole.ANALYST.name(), UserRole.ADMIN.name())

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