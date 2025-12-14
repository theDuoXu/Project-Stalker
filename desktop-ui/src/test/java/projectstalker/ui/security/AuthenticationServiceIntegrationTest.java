package projectstalker.ui.security;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.context.annotation.PropertySource;
import org.springframework.test.context.TestPropertySource;
import projectstalker.ui.StalkerUiLauncher;

import java.awt.Desktop;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import static org.assertj.core.api.Assertions.assertThat;

// Cargamos el contexto REAL de la aplicación (esto lee application.properties)
@SpringBootTest(classes = StalkerUiLauncher.class)
@Tag("Integration")
public class AuthenticationServiceIntegrationTest {

    @Autowired
    private AuthenticationService authService;

    @Test
    void testContextLoads() {
        // Simple smoke test: Si esto pasa, Spring ha leído application.properties
        // y ha inyectado las URLs en el servicio correctamente.
        assertThat(authService).isNotNull();
    }

    /**
     * TEST INTERACTIVO:
     * Este test abrirá tu navegador real. Debes loguearte manualmente.
     * El test esperará hasta 60 segundos a que completes el proceso.
     */
    @Test
    // Solo ejecutamos esto si hay entorno gráfico (evita fallos en CI/GitHub Actions)
    @EnabledIf("isDesktopSupported")
    void testManualLoginFlow() throws ExecutionException, InterruptedException, TimeoutException {
        System.out.println(">>> INICIANDO TEST INTERACTIVO DE LOGIN <<<");
        System.out.println("1. Se abrirá una ventana del navegador.");
        System.out.println("2. Por favor, inicia sesión en Keycloak.");
        System.out.println("3. El test esperará el token de vuelta.");

        // 1. Lanzamos el login (esto devuelve un CompletableFuture)
        var loginFuture = authService.login();

        // 2. Bloqueamos el test esperando tu acción (máximo 60 segundos)
        // Si no haces login en 5 minuto, el test falla.
        TokenResponse token = loginFuture.get(300, TimeUnit.SECONDS);

        // 3. Validaciones
        System.out.println(">>> LOGIN COMPLETADO CON ÉXITO <<<");
        System.out.println("Access Token recibido: " + token.accessToken().substring(0, 20) + "...");

        assertThat(token).isNotNull();
        assertThat(token.accessToken()).isNotBlank();
        assertThat(authService.isAuthenticated()).isTrue();
    }

    // Método de utilidad para JUnit Condition
    static boolean isDesktopSupported() {
        return Desktop.isDesktopSupported() && Desktop.getDesktop().isSupported(Desktop.Action.BROWSE);
    }
}