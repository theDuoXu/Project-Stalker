package projectstalker.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpMethod;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configurers.AbstractHttpConfigurer;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.oauth2.server.resource.authentication.JwtAuthenticationToken;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.core.convert.converter.Converter;
import org.springframework.security.oauth2.jwt.Jwt;

import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Configuration
@EnableWebSecurity
public class SecurityConfig {
    /**
     * Nuestro método de autentificación es invulnerable a Cross site request forgery porque utilizamos JWTs que no se
     * almacenan en una cookie (porque además ni siquiera es una web). Es imposible que una request pueda conseguir el token
     * debido a la política de Same origin policy y el navegador no lo adjunto al momento de enviar peticiones entre sitios.
     *
     * @param http Builder para SecurityFilter Chain
     * @return El bean que intercepta todas las peticiones Http
     * @throws Exception Excepción en la creación del interceptor
     */
    @Bean
    public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
        http
                .csrf(AbstractHttpConfigurer::disable)
                .authorizeHttpRequests(auth -> auth
                        // -----------------------------------------------------------
                        // 1. ACCESO PÚBLICO (Sin Token / Anónimo)
                        // -----------------------------------------------------------
                        // Documentación API y Swagger
                        .requestMatchers("/v3/api-docs/**", "/swagger-ui/**").permitAll()
                        // Documentos Legales (Privacy statements)
                        .requestMatchers("/api/public/legal/**", "/api/public/privacy").permitAll()
                        // Sensores Tiempo Real (SOLO lectura del estado actual, no históricos)
                        .requestMatchers(HttpMethod.GET, "/api/sensors/realtime", "/api/sensors/status").permitAll()

                        // -----------------------------------------------------------
                        // 2. ACCESO GUEST (Requiere Token con rol GUEST o superior)
                        // -----------------------------------------------------------
                        // Históricos de sensores y descarga de series temporales
                        .requestMatchers(HttpMethod.GET, "/api/sensors/*/history", "/api/sensors/export/**")
                        .hasAnyRole("GUEST", "TECHNICIAN", "ANALYST", "OFFICER", "ADMIN")

                        // -----------------------------------------------------------
                        // 3. ROLES OPERATIVOS (Technician, Analyst, Officer, Admin)
                        // -----------------------------------------------------------

                        // --- ADMIN (Gestión Total) ---
                        .requestMatchers("/api/admin/**").hasRole("ADMIN")

                        // --- GESTIÓN DE RÍOS Y SIMULACIONES (Escritura) ---
                        // Analyst y Admin crean/modifican
                        .requestMatchers(HttpMethod.POST, "/api/twins/**", "/api/sims/**").hasAnyRole("ANALYST", "ADMIN")
                        .requestMatchers(HttpMethod.PUT, "/api/twins/**").hasAnyRole("ANALYST", "ADMIN")
                        .requestMatchers(HttpMethod.DELETE, "/api/twins/**").hasRole("ADMIN")

                        // --- VISUALIZACIÓN DE SIMULACIONES E INFRAESTRUCTURA ---
                        // Technician puede ver configuración profunda y resultados, Guest NO
                        .requestMatchers(HttpMethod.GET, "/api/twins/**", "/api/sims/**").hasAnyRole("TECHNICIAN", "ANALYST", "ADMIN")

                        // --- ALERTAS (Gestión Operativa) ---
                        // Registrar alerta manual
                        .requestMatchers(HttpMethod.POST, "/api/alerts/manual").hasAnyRole("TECHNICIAN", "ANALYST", "ADMIN")
                        // Gestionar/Cerrar alertas (Officer)
                        .requestMatchers(HttpMethod.PUT, "/api/alerts/**").hasAnyRole("OFFICER", "ADMIN")

                        // --- INFORMES Y ACTUACIÓN ---
                        .requestMatchers("/api/reports/**").hasAnyRole("OFFICER", "ANALYST", "ADMIN")

                        // -----------------------------------------------------------
                        // 4. CANDADO FINAL
                        // -----------------------------------------------------------
                        .anyRequest().authenticated()
                )
                .oauth2ResourceServer(oauth2 -> oauth2
                        .jwt(jwt -> jwt.jwtAuthenticationConverter(jwtAuthenticationConverter()))
                )
                .sessionManagement(sess -> sess.sessionCreationPolicy(SessionCreationPolicy.STATELESS));

        return http.build();
    }

    /**
     * Este convertidor es VITAL.
     * Keycloak guarda los roles en "realm_access" -> "roles".
     * Spring por defecto busca en "scope".
     * Aquí extraemos los roles de Keycloak y les ponemos el prefijo "ROLE_".
     */
    private Converter<Jwt, JwtAuthenticationToken> jwtAuthenticationConverter() {
        return jwt -> {
            Map<String, Object> realmAccess = (Map<String, Object>) jwt.getClaims().get("realm_access");

            if (realmAccess == null || realmAccess.isEmpty()) {
                return new JwtAuthenticationToken(jwt, List.of());
            }

            Collection<String> roles = (Collection<String>) realmAccess.get("roles");

            var authorities = roles.stream()
                    .map(role -> new SimpleGrantedAuthority("ROLE_" + role)) // Transforma "ADMIN" en "ROLE_ADMIN"
                    .collect(Collectors.toList());

            // El "principal" es el nombre del usuario (preferred_username) en lugar del ID
            String principalName = jwt.getClaimAsString("preferred_username");

            return new JwtAuthenticationToken(jwt, authorities, principalName);
        };
    }
}