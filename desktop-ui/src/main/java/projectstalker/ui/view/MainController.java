package projectstalker.ui.view;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import javafx.application.Platform;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.ListView;
import javafx.scene.layout.StackPane;
import lombok.extern.slf4j.Slf4j;
import org.kordamp.ikonli.javafx.FontIcon;
import org.springframework.context.ApplicationContext;
import org.springframework.stereotype.Component;
import projectstalker.domain.dto.twin.TwinSummaryDTO;
import projectstalker.ui.security.AuthenticationService;
import projectstalker.ui.service.DigitalTwinClientService;
import projectstalker.ui.view.components.TwinListCell;

import java.io.IOException;
import java.util.Base64;

@Slf4j
@Component
public class MainController {

    private final AuthenticationService authService;
    private final DigitalTwinClientService twinService;
    private final ApplicationContext springContext; // <--- NECESARIO PARA CARGAR OTROS FXML
    private final ObjectMapper objectMapper = new ObjectMapper();

    @FXML public Label statusLabel;
    @FXML public Button loginButton;
    @FXML public FontIcon loginIcon;

    // Contenedor central dinámico
    @FXML public StackPane contentArea;

    @FXML public Button newProjectButton;

    // Vinculado al FXML
    @FXML public ListView<TwinSummaryDTO> projectListView;

    public MainController(AuthenticationService authService,
                          DigitalTwinClientService twinService,
                          ApplicationContext springContext) {
        this.authService = authService;
        this.twinService = twinService;
        this.springContext = springContext;
    }

    @FXML
    public void initialize() {
        statusLabel.setText("Sistema DSS Inicializado. Esperando autenticación...");

        // Configuramos la CellFactory separada
        projectListView.setCellFactory(listView -> new TwinListCell());

        // Desactivamos el botón de crear hasta que haya login
        newProjectButton.setDisable(true);
    }

    @FXML
    public void onLoginClick() {
        statusLabel.setText("Esperando autorización en el navegador...");
        loginButton.setDisable(true);
        attemptLogin();
    }

    @FXML
    public void onNewProjectClick() {
        try {
            // Cargamos el FXML del Editor
            FXMLLoader loader = new FXMLLoader(getClass().getResource("/fxml/river-editor.fxml"));

            // Le decimos al FXMLLoader que use Spring para crear el controlador (De la misma manera cargamos el MainView)
            // Esto permite que RiverEditorController tenga @Autowired o inyección por constructor.
            loader.setControllerFactory(springContext::getBean);

            Parent view = loader.load();

            // 3. Reemplazamos el contenido del centro
            contentArea.getChildren().clear();
            contentArea.getChildren().add(view);

        } catch (IOException e) {
            log.error(e.toString());
            showErrorAlert(new RuntimeException("No se pudo cargar la vista del editor: " + e.getMessage()));
        }
    }

    private void attemptLogin() {
        authService.login().thenAccept(token -> {
            String username = extractUsername(token.accessToken());

            Platform.runLater(() -> {
                statusLabel.setText("CONECTADO | Usuario: " + username);
                statusLabel.setStyle("-fx-text-fill: -color-success-fg; -fx-font-weight: bold;");

                loginButton.setText("Cerrar Sesión");
                loginButton.setDisable(false);
                if (loginIcon != null) loginIcon.setIconLiteral("mdi2a-account-check");

                loginButton.setOnAction(e -> doLogout());

                // Habilitamos creación
                newProjectButton.setDisable(false);

                // Cargar proyectos tras login exitoso
                loadProjects();
            });

        }).exceptionally(ex -> {
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

    public void loadProjects() {
        statusLabel.setText("Cargando catálogo de ríos...");

        twinService.getAllTwins()
                .collectList()
                .subscribe(projects -> {
                    Platform.runLater(() -> {
                        projectListView.getItems().setAll(projects);
                        statusLabel.setText("Proyectos cargados: " + projects.size());
                    });
                }, error -> {
                    Platform.runLater(() -> {
                        statusLabel.setText("Error obteniendo datos del servidor.");
                    });
                });
    }

    private void doLogout() {
        log.info("Iniciando cierre de sesión en Keycloak...");
        loginButton.setDisable(true);
        statusLabel.setText("Cerrando sesión en Keycloak...");

        // Llamamos al servicio de autenticación para cerrar la sesión externa
        authService.logout()
                .exceptionally(ex -> {
                    log.error("Fallo al contactar Keycloak para cerrar sesión: {}", ex.getMessage());
                    // Permite que la desconexión local continúe incluso si el cierre de sesión remoto falla
                    return null;
                })
                .thenRun(() -> {
                    // Se ejecuta sin importar si el logout remoto falló o tuvo éxito
                    Platform.runLater(() -> {
                        // --- LIMPIEZA DE UI (solo se ejecuta tras el logout asíncrono) ---
                        statusLabel.setText("Desconectado.");
                        statusLabel.setStyle("");
                        loginButton.setText("Conectar a Keycloak");
                        loginButton.setDisable(false); // Habilitamos para iniciar sesión de nuevo
                        if (loginIcon != null) loginIcon.setIconLiteral("mdi2l-login");

                        // Limpiamos la lista y el panel central al salir
                        projectListView.getItems().clear();
                        contentArea.getChildren().clear();
                        contentArea.getChildren().add(new Label("Selecciona o crea un Gemelo Digital para comenzar"));

                        newProjectButton.setDisable(true);

                        // Restaurar el evento al modo de login
                        loginButton.setOnAction(e -> onLoginClick());
                    });
                });
    }
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
        alert.setTitle("Error");
        alert.setHeaderText("Ha ocurrido un problema");
        alert.setContentText(ex.getCause() != null ? ex.getCause().getMessage() : ex.getMessage());
        alert.showAndWait();
    }
}