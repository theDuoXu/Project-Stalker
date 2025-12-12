package projectstalker.compute.api;

import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.AccessDeniedException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import projectstalker.domain.exception.InvalidExportRequestException;
import projectstalker.domain.exception.ResourceNotFoundException;
import projectstalker.domain.exception.SensorBusinessException;

import java.time.LocalDateTime;
import java.util.Map;

@Slf4j // <-- Lombok inyecta automáticamente la variable 'log'
@RestControllerAdvice
public class GlobalExceptionHandler {

    /**
     * Maneja errores de lógica de negocio (ej: pedir export "ALL", sensor desconocido).
     * Log: WARN (No es un error del sistema, es un error del cliente/petición).
     */
    @ExceptionHandler({SensorBusinessException.class, InvalidExportRequestException.class})
    public ResponseEntity<Object> handleSensorBusinessException(SensorBusinessException ex) {
        // Logueamos solo el mensaje, no el stacktrace, para no ensuciar los logs
        log.warn("Business Rule Violation: {}", ex.getMessage());

        return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(Map.of(
                "timestamp", LocalDateTime.now(),
                "status", 400,
                "error", "Invalid Sensor Request",
                "message", ex.getMessage()
        ));
    }

    /**
     * Maneja errores de lógica de negocio de gemelos digitales.
     * Log: WARN (No es un error del sistema, es un error del cliente/petición).
     */
    @ExceptionHandler({ResourceNotFoundException.class})
    public ResponseEntity<Object> handleTwinBusinessException(SensorBusinessException ex) {
        // Logueamos solo el mensaje, no el stacktrace, para no ensuciar los logs
        log.warn("Business Rule Violation: {}", ex.getMessage());

        return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(Map.of(
                "timestamp", LocalDateTime.now(),
                "status", 404,
                "error", "Twin not found",
                "message", ex.getMessage()
        ));
    }

    /**
     * Maneja errores de seguridad (ej: GUEST pidiendo 10 días de datos).
     * Log: WARN (Intento de acceso no autorizado).
     */
    @ExceptionHandler(AccessDeniedException.class)
    public ResponseEntity<Object> handleAccessDenied(AccessDeniedException ex) {
        log.warn("Security Alert: Access Denied to requested resource. Reason: {}", ex.getMessage());

        return ResponseEntity.status(HttpStatus.FORBIDDEN).body(Map.of(
                "timestamp", LocalDateTime.now(),
                "status", 403,
                "error", "Forbidden",
                "message", "You do not have permission to access this resource or the requested scope is too broad."
        ));
    }

    /**
     * Maneja todo lo demás (NullPointer, SQL connection failed, etc.).
     * Log: ERROR (Incluye StackTrace completo).
     */
    @ExceptionHandler(Exception.class)
    public ResponseEntity<Object> handleGeneralErrors(Exception ex) {
        // Aquí SÍ pasamos la excepción 'ex' completa para ver la traza del error en consola/archivo
        log.error("Unexpected System Error occurred", ex);

        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(Map.of(
                "timestamp", LocalDateTime.now(),
                "status", 500,
                "error", "Internal Server Error",
                "message", "An unexpected error occurred. Please contact support referencing this timestamp."
        ));
    }
}