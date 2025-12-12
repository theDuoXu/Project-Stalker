package projectstalker.security;

import org.springframework.security.core.Authentication;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.Collection;
import java.util.Set;

@Component("sensorExportValidator") // Nombre del bean para usar en Spring Expression Language
public class ExportAccessValidator {

    private static final String ROLE_ADMIN = "ROLE_ADMIN";
    private static final String ROLE_GUEST = "ROLE_GUEST";
    // Agrupamos los roles operativos para no repetir código
    private static final Set<String> STAFF_ROLES = Set.of(
            "ROLE_TECHNICIAN", "ROLE_ANALYST", "ROLE_OFFICER"
    );

    /**
     * Valida si el usuario tiene permiso para exportar el rango de fechas solicitado.
     */
    public boolean canExport(Authentication authentication, LocalDateTime from, LocalDateTime to) {
        // 1. Si no hay autenticación, denegar
        if (authentication == null || !authentication.isAuthenticated()) {
            return false; // No debería pasar porque Security Config protege primero
        }

        Collection<? extends GrantedAuthority> authorities = authentication.getAuthorities();
        boolean isAdmin = hasRole(authorities, ROLE_ADMIN);

        // 2. REGLA SUPREMA: Admin puede hacer lo que quiera (incluso rangos nulos/infinitos)
        if (isAdmin) {
            return true;
        }

        // 3. Validación de Fechas Nulas (Solo Admin puede pedir "toda la historia")
        if (from == null || to == null) {
            return false; // Si no es admin y faltan fechas, denegado.
        }

        // 4. Calcular duración en días
        long daysRequested = ChronoUnit.DAYS.between(from, to);
        if (daysRequested < 0) {
            return false; // Fechas invertidas o error
        }
        LocalDateTime now = LocalDateTime.now();

        // 5. Aplicar reglas por Rol, de menos a más

        // Verificamos si es STAFF (Technician, Analyst, Officer)
        boolean isStaff = authorities.stream()
                .anyMatch(a -> STAFF_ROLES.contains(a.getAuthority()));

        if (isStaff) {
            // STAFF: Máximo 30 días, 90 días de antigüedad
            return daysRequested <= 30 && ChronoUnit.DAYS.between(from, now) <= 90;
        }

        boolean isGuest = hasRole(authorities, ROLE_GUEST);

        if (isGuest) {
            // GUEST: Máximo 1 día y máximo rango de visibilidad una semana
            // Como ya hemos validado que from tiene que ser anterior a to
            return daysRequested <= 1 && ChronoUnit.DAYS.between(from, now) <= 7;
        }

        // Si tiene un rol raro que no controlamos
        return false;
    }

    private boolean hasRole(Collection<? extends GrantedAuthority> authorities, String role) {
        return authorities.stream().anyMatch(a -> a.getAuthority().equals(role));
    }
}