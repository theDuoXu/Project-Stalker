package projectstalker.ui.view;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import javafx.application.Platform;
import javafx.fxml.FXML;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import org.kordamp.ikonli.javafx.FontIcon;
import org.springframework.stereotype.Component;
import projectstalker.ui.security.AuthenticationService;

import java.util.Base64;

@Component
public class MainController {

    private final AuthenticationService authService;
    private final ObjectMapper objectMapper = new ObjectMapper(); // Para leer el payload del JWT

    @FXML
    public Label statusLabel;

    @FXML
    public Button loginButton;

    @FXML
    public FontIcon loginIcon;

    public MainController(AuthenticationService authService) {
        this.authService = authService;
    }

    @FXML
    public void initialize() {
        statusLabel.setText("Sistema DSS Inicializado. Esperando autenticación...");
    }

    @FXML
    public void onLoginClick() {
        // 1. Estado visual de "Cargando"
        statusLabel.setText("Esperando autorización en el navegador...");
        loginButton.setDisable(true);

        // 2. Llamada al servicio (Asíncrona)
        attemptLogin();
    }

    private void attemptLogin() {
        authService.login().thenAccept(token -> {

            // Extraer usuario del token para mostrarlo (Decodificación simple de JWT)
            String username = extractUsername(token.accessToken());

            // 3. Volver al hilo de JavaFX para tocar la UI
            Platform.runLater(() -> {
                statusLabel.setText("CONECTADO | Usuario: " + username);
                statusLabel.setStyle("-fx-text-fill: -color-success-fg; -fx-font-weight: bold;");

                loginButton.setText("Cerrar Sesión");
                loginButton.setDisable(false);
                // Cambiar icono a 'User'
                if (loginIcon != null) loginIcon.setIconLiteral("mdi2a-account-check");

                // Cambiar la acción del botón para hacer logout la próxima vez
                loginButton.setOnAction(e -> doLogout());
            });

        }).exceptionally(ex -> {
            // Manejo de errores en el hilo de JavaFX
            Platform.runLater(() -> {
                statusLabel.setText("Error: " + ex.getMessage());
                statusLabel.setStyle("-fx-text-fill: -color-danger-fg;");
                loginButton.setDisable(false);
                loginButton.setText("Reintentar Conexión");

                showErrorAlert(ex);
            });
            return null;
        });
    }

    private void doLogout() {
        // Implementación simple de logout visual
        statusLabel.setText("Desconectado.");
        statusLabel.setStyle("");
        loginButton.setText("Conectar a Keycloak");
        if (loginIcon != null) loginIcon.setIconLiteral("mdi2l-login");
        loginButton.setOnAction(e -> onLoginClick());
    }

    // Utilidad rápida para leer el "preferred_username" del JWT sin validar firma (el backend valida)
    private String extractUsername(String accessToken) {
        try {
            String[] parts = accessToken.split("\\.");
            if (parts.length < 2) return "Usuario";

            String payload = new String(Base64.getUrlDecoder().decode(parts[1]));
            JsonNode node = objectMapper.readTree(payload);

            if (node.has("preferred_username")) {
                return node.get("preferred_username").asText();
            }
            return node.has("sub") ? node.get("sub").asText() : "Usuario";

        } catch (Exception e) {
            return "Usuario (Token Opaque)";
        }
    }

    private void showErrorAlert(Throwable ex) {
        Alert alert = new Alert(Alert.AlertType.ERROR);
        alert.setTitle("Error de Autenticación");
        alert.setHeaderText("No se pudo conectar con Keycloak");
        alert.setContentText(ex.getCause() != null ? ex.getCause().getMessage() : ex.getMessage());
        alert.showAndWait();
    }
}