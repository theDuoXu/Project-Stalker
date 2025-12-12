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
                                "/v3/api-docs/**",
                                "/swagger-ui/**",
                                "/api/public/**"
                        ).permitAll()

                        // 2. SENSORES (Lectura Pública vs Histórico Protegido)
                        .requestMatchers(HttpMethod.GET, "/api/sensors/*/realtime", "/api/sensors/*/status").permitAll()
                        .requestMatchers(HttpMethod.GET, "/api/sensors/*/history", "/api/sensors/export/**")
                        .hasAnyRole(GUEST, TECHNICIAN, ANALYST, OFFICER, ADMIN)

                        // 3. ADMIN GLOBAL
                        .requestMatchers("/api/admin/**").hasRole(ADMIN)

                        // 4. SIMULACIONES & DIGITAL TWINS
                        .requestMatchers(HttpMethod.POST, "/api/twins/**", "/api/sims/**").hasAnyRole(ANALYST, ADMIN)
                        .requestMatchers(HttpMethod.PUT, "/api/twins/**").hasAnyRole(ANALYST, ADMIN)
                        .requestMatchers(HttpMethod.DELETE, "/api/twins/**").hasRole(ADMIN)
                        .requestMatchers(HttpMethod.GET, "/api/twins/**", "/api/sims/**").hasAnyRole(TECHNICIAN, ANALYST, ADMIN)

                        // 5. OPERACIONES (Alertas e Informes)
                        .requestMatchers(HttpMethod.POST, "/api/alerts/manual").hasAnyRole(TECHNICIAN, ANALYST, ADMIN)
                        .requestMatchers(HttpMethod.PUT, "/api/alerts/**").hasAnyRole(OFFICER, ADMIN)
                        .requestMatchers("/api/reports/**").hasAnyRole(OFFICER, ANALYST, ADMIN)

                        // 6. DEFAULT
                        .anyRequest().authenticated()
                )

                // Configuración OAuth2 Limpia
                .oauth2ResourceServer(oauth2 -> oauth2
                        .jwt(jwt -> jwt.jwtAuthenticationConverter(keycloakJwtConverter))
                );

        return http.build();
    }
}