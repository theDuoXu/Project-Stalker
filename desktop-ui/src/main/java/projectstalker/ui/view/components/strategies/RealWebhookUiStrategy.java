package projectstalker.ui.view.components.strategies;

import javafx.scene.Node;
import javafx.scene.control.Label;
import javafx.scene.control.PasswordField;
import javafx.scene.control.TextField;
import javafx.scene.layout.VBox;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.Map;

@Component
public class RealWebhookUiStrategy implements SensorUiStrategy {

    private final VBox container = new VBox(10);
    private final TextField urlField = new TextField();
    private final PasswordField tokenField = new PasswordField();

    public RealWebhookUiStrategy() {
        urlField.setPromptText("https://api.device-network.com/v1/readings");

        javafx.scene.control.Button testBtn = new javafx.scene.control.Button("Probar Conexión");
        testBtn.setOnAction(e -> testConnection(testBtn));

        container.getChildren().addAll(
                new Label("Endpoint URL (Webhook):"), urlField,
                new Label("Auth Bearer Token:"), tokenField,
                testBtn);
    }

    private void testConnection(javafx.scene.control.Button btn) {
        String url = urlField.getText();
        if (url.isBlank()) {
            new javafx.scene.control.Alert(javafx.scene.control.Alert.AlertType.WARNING, "URL vacía").showAndWait();
            return;
        }

        btn.setDisable(true);
        btn.setText("Conectando...");

        new Thread(() -> {
            try {
                java.net.http.HttpClient client = java.net.http.HttpClient.newHttpClient();
                java.net.http.HttpRequest request = java.net.http.HttpRequest.newBuilder()
                        .uri(java.net.URI.create(url))
                        .GET()
                        .timeout(java.time.Duration.ofSeconds(5))
                        .build();

                java.net.http.HttpResponse<String> response = client.send(request,
                        java.net.http.HttpResponse.BodyHandlers.ofString());

                javafx.application.Platform.runLater(() -> {
                    btn.setDisable(false);
                    btn.setText("Probar Conexión");
                    String msg = "Status: " + response.statusCode() + "\nBody: " + response.body();
                    javafx.scene.control.Alert.AlertType type = response.statusCode() == 200
                            ? javafx.scene.control.Alert.AlertType.INFORMATION
                            : javafx.scene.control.Alert.AlertType.WARNING;
                    new javafx.scene.control.Alert(type, msg).showAndWait();
                });
            } catch (Exception e) {
                javafx.application.Platform.runLater(() -> {
                    btn.setDisable(false);
                    btn.setText("Probar Conexión");
                    new javafx.scene.control.Alert(javafx.scene.control.Alert.AlertType.ERROR,
                            "Error: " + e.getMessage()).showAndWait();
                });
            }
        }).start();
    }

    @Override
    public String getDisplayName() {
        return "Real: Conexión HTTP/Webhook";
    }

    @Override
    public String getStrategyCode() {
        return "REAL_IoT_WEBHOOK";
    }

    @Override
    public SensorStrategyCategory getCategory() {
        return SensorStrategyCategory.REAL;
    }

    @Override
    public Node render() {
        return container;
    }

    @Override
    public boolean validate() {
        return !urlField.getText().isBlank();
    }

    @Override
    public Map<String, Object> extractConfiguration() {
        Map<String, Object> map = new HashMap<>();
        map.put("url", urlField.getText());
        map.put("token", tokenField.getText());
        return map;
    }

    @Override
    public void reset() {
        urlField.clear();
        tokenField.clear();
    }

    @Override
    public void populate(Map<String, Object> config) {
        if (config != null) {
            if (config.containsKey("url"))
                urlField.setText(String.valueOf(config.get("url")));
            if (config.containsKey("token"))
                tokenField.setText(String.valueOf(config.get("token")));
        }
    }
}