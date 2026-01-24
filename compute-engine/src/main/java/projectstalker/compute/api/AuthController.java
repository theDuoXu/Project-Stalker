package projectstalker.compute.api;

import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.oauth2.jwt.Jwt;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import projectstalker.compute.api.dto.UserDTO;

import java.util.List;
import java.util.Map;
import java.util.UUID;

@RestController
@RequestMapping("/auth")
public class AuthController {

    @GetMapping("/me")
    public UserDTO getCurrentUser(@AuthenticationPrincipal Jwt jwt) {
        if (jwt == null) {
            // Should be handled by Security filter, but just in case
            throw new RuntimeException("No authentication token found");
        }

        String idStr = jwt.getClaimAsString("sub");
        String username = jwt.getClaimAsString("preferred_username");
        String email = jwt.getClaimAsString("email");

        // Determine role from Keycloak claims
        // Default to 'operador'
        String role = "operador";

        // Extract realm_access.roles
        Map<String, Object> realmAccess = jwt.getClaim("realm_access");
        if (realmAccess != null && realmAccess.containsKey("roles")) {
            @SuppressWarnings("unchecked")
            List<String> roles = (List<String>) realmAccess.get("roles");
            if (roles != null) {
                if (roles.contains("admin") || roles.contains("administrador") || roles.contains("PROJECT_ADMIN")) {
                    role = "administrador";
                }
            }
        }

        UUID id = null;
        try {
            id = UUID.fromString(idStr);
        } catch (IllegalArgumentException e) {
            // Fallback or handle null
            id = UUID.randomUUID(); // Should not happen in prod with valid setup
        }

        return new UserDTO(id, username, email, role);
    }
}
