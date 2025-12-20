package projectstalker.ui.view;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.javafx.scene.control.LabeledText;
import javafx.animation.KeyFrame;
import javafx.animation.KeyValue;
import javafx.animation.Timeline;
import javafx.application.Platform;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.ListView;
import javafx.scene.input.MouseButton;
import javafx.scene.input.MouseEvent;
import javafx.scene.layout.Region;
import javafx.scene.layout.StackPane;
import javafx.scene.layout.VBox;
import javafx.util.Duration;
import lombok.extern.slf4j.Slf4j;
import org.kordamp.ikonli.javafx.FontIcon;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.context.ApplicationContext;
import org.springframework.context.ApplicationEventPublisher;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Component;
import projectstalker.domain.dto.twin.TwinSummaryDTO;
import projectstalker.ui.event.*;
import projectstalker.ui.security.AuthenticationService;
import projectstalker.ui.service.DigitalTwinClientService;
import projectstalker.ui.view.components.TwinListCell;
import projectstalker.ui.view.dashboard.TwinDashboardController;
import projectstalker.ui.viewmodel.StatusTarget;
import projectstalker.ui.viewmodel.StatusType;
import projectstalker.ui.viewmodel.StatusViewModel;

import java.io.IOException;
import java.util.Base64;

@Slf4j
@Component
public class MainController {

    private final AuthenticationService authService;
    private final DigitalTwinClientService twinService;
    private final ApplicationContext springContext;
    private final ObjectMapper objectMapper = new ObjectMapper();
    private final ApplicationEventPublisher eventPublisher;
    private final StatusViewModel statusViewModel;
    private final StatusViewModel hpcStatusViewModel;

    @FXML public Label statusLabel;
    @FXML public Label hpcStatusLabel;
    @FXML public Button loginButton;
    @FXML public FontIcon loginIcon;

    @FXML public StackPane contentArea;
    @FXML public VBox twinsSideBar;
    @FXML public Button newProjectButton;
    @FXML public ListView<TwinSummaryDTO> projectListView;

    private double originalSidebarWidth = 250.0;

    // Guardamos el string base de conexión para restaurarlo tras cargas temporales
    private String connectionStatusString = "Desconectado";

    public MainController(AuthenticationService authService,
                          DigitalTwinClientService twinService,
                          ApplicationContext springContext,
                          ApplicationEventPublisher eventPublisher,
                          @Qualifier("mainStatusViewModel") StatusViewModel statusViewModel,
                          @Qualifier("hpcStatusViewModel") StatusViewModel hpcStatusViewModel) {
        this.authService = authService;
        this.twinService = twinService;
        this.springContext = springContext;
        this.eventPublisher = eventPublisher;
        this.statusViewModel = statusViewModel;
        this.hpcStatusViewModel = hpcStatusViewModel;
    }

    @FXML
    public void initialize() {
        // 1. Vinculación reactiva MAIN
        statusLabel.textProperty().bind(this.statusViewModel.statusMessageProperty());
        this.statusViewModel.statusTypeProperty().addListener((obs, oldVal, newVal) ->
                applyStyle(statusLabel, newVal)
        );

        // 1b. Vinculación reactiva HPC
        hpcStatusLabel.textProperty().bind(this.hpcStatusViewModel.statusMessageProperty());
        this.hpcStatusViewModel.statusTypeProperty().addListener((obs, oldVal, newVal) ->
                applyStyle(hpcStatusLabel, newVal)
        );

        // 2. Estado inicial vía evento
        publishPermanentStatus("Sistema DSS Inicializado... Esperando login");
        publishPermanentStatus("HPC: Desconectado", StatusType.DEFAULT, StatusTarget.HPC);

        setupListView();

        newProjectButton.setDisable(true);

        Platform.runLater(() -> {
            if (twinsSideBar.getWidth() > 0) {
                originalSidebarWidth = twinsSideBar.getWidth();
            }
        });
    }

    private void setupListView() {
        // 1. CONFIGURACIÓN DE LAS CELDAS
        projectListView.setCellFactory(listView -> {
            TwinListCell cell = new TwinListCell();

            // Usamos setOnMouseClicked para Single y Double click
            // Esto garantiza que la selección nativa de JavaFX ya ocurrió (en el MousePressed)
            // antes de que nosotros hagamos cambios en la UI.
            cell.setOnMouseClicked(event -> {
                // Seguridad: Si la celda está vacía, no hacemos nada
                if (cell.isEmpty() || cell.getItem() == null) {
                    return;
                }

                // Solo botón izquierdo
                if (event.getButton() == MouseButton.PRIMARY) {

                    // CASO A: Doble Click -> Editor
                    if (event.getClickCount() == 2) {
                        log.info("Doble click: Abriendo Editor");
                        openEditorForTwin(cell.getItem());

                        // CASO B: Click Simple -> Dashboard
                    } else if (event.getClickCount() == 1) {
                        log.info("Click simple: Abriendo Dashboard");
                        openDashBoardForTwin(cell.getItem());
                    }
                    event.consume();
                }
            });

            return cell;
        });

        // 2. DESELECCIONAR AL PULSAR EN VACÍO
        projectListView.setOnMouseClicked(event -> {
            // Si el click fue consumido por la celda (arriba), el target no llegará aquí igual.
            // Pero por seguridad verificamos:

            if (event.getButton() == MouseButton.PRIMARY && event.getClickCount() == 1) {
                // El truco es que si diste en una celda, el evento lo manejó 'cell.setOnMouseClicked'
                // Pero si diste en el fondo, JavaFX propaga el evento hasta aquí.

                // Verificamos si hay algún item seleccionado y el click no fue en una celda
                // (Una forma "barata" es ver si el target es la propia ListView o el contenedor virtual)
                if (event.getTarget() instanceof ListView ||
                        event.getTarget() instanceof Region) { // flow

                    projectListView.getSelectionModel().clearSelection();
                }
            }
        });
    }

    private void openDashBoardForTwin(TwinSummaryDTO twin) {
        log.info("Abriendo Dashboard para: {}", twin.name());
        try {
            FXMLLoader loader = new FXMLLoader(getClass().getResource("/fxml/dashboard/twin-dashboard.fxml"));

            loader.setControllerFactory(springContext::getBean);

            Parent view = loader.load();

            // Obtener el controlador principal del Dashboard y pasarle los datos
            TwinDashboardController controller = loader.getController();
            controller.setTwin(twin);

            // Renderizar en el área central
            contentArea.getChildren().clear();
            contentArea.getChildren().add(view);

            // Ocultar la barra lateral para dar protagonismo al Dashboard
            toggleSidebar(false);

        } catch (IOException e) {
            log.error("Error fatal cargando el Dashboard", e);
            showErrorAlert(new RuntimeException("No se pudo cargar la vista del Dashboard. \n" + e.getMessage()));
        }
    }

    private void openEditorForTwin(TwinSummaryDTO twin) {
        try {
            FXMLLoader loader = new FXMLLoader(getClass().getResource("/fxml/river-editor.fxml"));
            loader.setControllerFactory(springContext::getBean);
            Parent view = loader.load();

            // Configurar el controlador en modo edición
            RiverEditorController controller = loader.getController();
            controller.setEditingTwin(twin);

            contentArea.getChildren().clear();
            contentArea.getChildren().add(view);
            toggleSidebar(false);
        } catch (IOException e) {
            log.error("Error abriendo editor", e);
        }
    }

    private void applyStyle(Label label, StatusType type) {
        label.getStyleClass().removeAll("status-error", "status-success", "status-warning");
        switch (type) {
            case SUCCESS -> label.getStyleClass().add("status-success");
            case ERROR -> label.getStyleClass().add("status-error");
            case WARNING -> label.getStyleClass().add("status-warning");
        }
    }

    @FXML
    public void onLoginClick() {
        // Estado permanente mientras dura el proceso de login
        publishPermanentStatus("Esperando autorización en el navegador...");
        loginButton.setDisable(true);
        attemptLogin();
    }

    @FXML
    public void onNewProjectClick() {
        try {
            FXMLLoader loader = new FXMLLoader(getClass().getResource("/fxml/river-editor.fxml"));
            loader.setControllerFactory(springContext::getBean);
            Parent view = loader.load();

            contentArea.getChildren().clear();
            contentArea.getChildren().add(view);
            toggleSidebar(false);
        } catch (IOException e) {
            log.error(e.toString());
            showErrorAlert(new RuntimeException("No se pudo cargar la vista del editor: " + e.getMessage()));
        }
    }

    @EventListener
    public void onSidebarEvent(SidebarVisibilityEvent event) {
        Platform.runLater(() -> toggleSidebar(event.visible()));
    }

    @EventListener
    public void onRestoreMainViewEvent(RestoreMainViewEvent event){
        Platform.runLater(() -> {
            contentArea.getChildren().clear();
            Label placeholder = new Label("Selecciona o crea un Gemelo Digital para comenzar");
            placeholder.setStyle("-fx-text-fill: -color-fg-muted;");
            contentArea.getChildren().add(placeholder);
        });
    }

    @EventListener
    public void onTwinListRefresh(TwinListRefreshEvent event) {
        log.info("Evento de refresco recibido. Recargando lista de proyectos...");
        loadProjects();
    }

    private void toggleSidebar(boolean show) {
        if (show) {
            twinsSideBar.setVisible(true);
            twinsSideBar.setManaged(true);
        }

        double targetWidth = show ? originalSidebarWidth : 0;

        Timeline timeline = new Timeline();
        timeline.getKeyFrames().add(new KeyFrame(Duration.millis(300),
                new KeyValue(twinsSideBar.prefWidthProperty(), targetWidth),
                new KeyValue(twinsSideBar.minWidthProperty(), targetWidth),
                new KeyValue(twinsSideBar.maxWidthProperty(), targetWidth),
                new KeyValue(twinsSideBar.opacityProperty(), show ? 1.0 : 0.0)
        ));

        timeline.setOnFinished(e -> {
            if (!show) {
                twinsSideBar.setVisible(false);
                twinsSideBar.setManaged(false);
            }
        });

        timeline.play();
    }

    private void attemptLogin() {
        authService.login().thenAccept(token -> {
            String username = extractUsername(token.accessToken());
            this.connectionStatusString = "CONECTADO | Usuario: " + username;

            Platform.runLater(() -> {
                // Actualizamos estado permanente de éxito
                publishPermanentStatus(connectionStatusString);
                publishTransitoryStatus(connectionStatusString, StatusViewModel.TransitionTime.MEDIUM ,StatusType.SUCCESS);

                loginButton.setText("Cerrar Sesión");
                loginButton.setDisable(false);
                if (loginIcon != null) loginIcon.setIconLiteral("mdi2a-account-check");

                loginButton.setOnAction(e -> doLogout());
                newProjectButton.setDisable(false);
                eventPublisher.publishEvent(new SidebarVisibilityEvent(true));

                loadProjects();
            });

        }).exceptionally(ex -> {
            Platform.runLater(() -> {
                // Error Transitorio (6 seg) y luego vuelve al estado anterior (Esperando login)
                publishTransitoryStatus(
                        "Error de conexión: " + ex.getMessage(),
                        StatusViewModel.TransitionTime.MEDIUM,
                        StatusType.ERROR);

                // Restauramos el botón
                loginButton.setDisable(false);
                loginButton.setText("Reintentar Conexión");
                showErrorAlert(ex);
            });
            return null;
        });
    }

    public void loadProjects() {
        // Cambiamos a estado permanente "Cargando" para que no desaparezca si tarda

        // --- AQUÍ USAMOS EL NUEVO VIEW MODEL DE HPC VÍA EVENTO ---
        publishPermanentStatus("Sincronizando...", StatusType.DEFAULT, StatusTarget.HPC);

        twinService.getAllTwins()
                .collectList()
                .subscribe(projects -> {
                    Platform.runLater(() -> {
                        projectListView.getItems().setAll(projects);

                        publishPermanentStatus("HPC: Desconectado", StatusType.DEFAULT, StatusTarget.HPC);

                        publishPermanentStatus(connectionStatusString);
                        publishTransitoryStatus("Proyectos cargados: " + projects.size(), StatusViewModel.TransitionTime.IMMEDIATE);
                        publishTransitoryStatus(connectionStatusString, StatusViewModel.TransitionTime.IMMEDIATE ,StatusType.SUCCESS);
                    });
                }, error -> {
                    Platform.runLater(() -> {
                        // Mantenemos el estado de error visible un tiempo razonable
                        publishTransitoryStatus("Error obteniendo datos del servidor.", StatusViewModel.TransitionTime.MEDIUM);

                        // --- ERROR EN HPC VÍA EVENTO ---
                        publishPermanentStatus("HPC Engine: DISCONNECTED", StatusType.ERROR, StatusTarget.HPC);
                    });
                });
    }

    private void doLogout() {
        log.info("Iniciando cierre de sesión en Keycloak...");
        loginButton.setDisable(true);
        publishPermanentStatus("Cerrando sesión en Keycloak...");

        authService.logout()
                .exceptionally(ex -> {
                    log.error("Fallo al contactar Keycloak: {}", ex.getMessage());
                    return null;
                })
                .thenRun(() -> {
                    Platform.runLater(() -> {
                        // Limpieza UI
                        statusLabel.setStyle(""); // Reset estilo
                        publishPermanentStatus("Desconectado.");

                        // Reset HPC
                        publishPermanentStatus("HPC: Desconectado", StatusType.DEFAULT, StatusTarget.HPC);

                        loginButton.setText("Conectar a Keycloak");
                        loginButton.setDisable(false);
                        if (loginIcon != null) loginIcon.setIconLiteral("mdi2l-login");

                        projectListView.getItems().clear();
                        contentArea.getChildren().clear();
                        contentArea.getChildren().add(new Label("Selecciona o crea un Gemelo Digital para comenzar"));

                        newProjectButton.setDisable(true);
                        loginButton.setOnAction(e -> onLoginClick());
                        eventPublisher.publishEvent(new SidebarVisibilityEvent(true));
                    });
                });
    }

    // --- Helpers de Eventos para reducir verbosidad ---

    private void publishPermanentStatus(String message) {
        eventPublisher.publishEvent(new PermanentStatusUpdateEvent(message, StatusType.DEFAULT, StatusTarget.APP));
    }

    private void publishPermanentStatus(String message, StatusType statusType, StatusTarget target) {
        eventPublisher.publishEvent(new PermanentStatusUpdateEvent(message, statusType, target));
    }

    private void publishPermanentStatus(String message, StatusType statusType) {
        publishPermanentStatus(message, statusType, StatusTarget.APP);
    }

    private void publishTransitoryStatus(String message, StatusViewModel.TransitionTime time) {
        eventPublisher.publishEvent(new TransitoryStatusUpdateEvent(message, StatusType.DEFAULT, time, StatusTarget.APP));
    }

    private void publishTransitoryStatus(String message, StatusViewModel.TransitionTime time, StatusType statusType) {
        eventPublisher.publishEvent(new TransitoryStatusUpdateEvent(message, statusType, time, StatusTarget.APP));
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