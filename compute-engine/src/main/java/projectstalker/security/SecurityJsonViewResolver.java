package projectstalker.security;

import org.springframework.core.MethodParameter;
import org.springframework.http.MediaType;
import org.springframework.http.converter.json.MappingJacksonValue;
import org.springframework.http.server.ServerHttpRequest;
import org.springframework.http.server.ServerHttpResponse;
import org.springframework.security.authentication.AnonymousAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.servlet.mvc.method.annotation.AbstractMappingJacksonResponseBodyAdvice;
import projectstalker.domain.sensors.SensorViews;

@RestControllerAdvice
public class SecurityJsonViewResolver extends AbstractMappingJacksonResponseBodyAdvice {

    // Este método se ejecuta JUSTO ANTES de escribir el JSON
    @Override
    protected void beforeBodyWriteInternal(
            MappingJacksonValue bodyContainer, // El contenedor del objeto a serializar
            MediaType contentType,
            MethodParameter returnType,
            ServerHttpRequest request,
            ServerHttpResponse response) {

        // 1. Miramos quién es el usuario
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();

        // 2. Lógica de decisión
        if (isAuthenticated(auth)) {
            // Si es un usuario real (Admin, Guest, etc.), aplicamos la vista INTERNAL
            // Verá: @JsonView(Public) Y @JsonView(Internal)
            bodyContainer.setSerializationView(SensorViews.Internal.class);
        } else {
            // Si es público/anónimo, aplicamos la vista PUBLIC
            // Verá: SOLO @JsonView(Public). Lo demás se borra del JSON.
            bodyContainer.setSerializationView(SensorViews.Public.class);
        }
    }

    private boolean isAuthenticated(Authentication auth) {
        return auth != null &&
                auth.isAuthenticated() &&
                !(auth instanceof AnonymousAuthenticationToken);
    }
}