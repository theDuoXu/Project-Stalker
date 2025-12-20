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
        container.getChildren().addAll(
                new Label("Endpoint URL (Webhook):"), urlField,
                new Label("Auth Bearer Token:"), tokenField
        );
    }

    @Override
    public String getDisplayName() { return "Real: Conexión HTTP/Webhook"; }

    @Override
    public String getStrategyCode() { return "REAL_IoT_WEBHOOK"; }

    @Override
    public boolean isVirtual() { return false; }

    @Override
    public Node render() { return container; }

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
    public void reset() { urlField.clear(); tokenField.clear(); }
}