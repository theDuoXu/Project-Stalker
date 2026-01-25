package projectstalker.ui.view.components.strategies;

import javafx.scene.Node;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.layout.VBox;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.Map;

@Component
public class ManualStaticUiStrategy implements SensorUiStrategy {

    private final VBox container = new VBox(10);
    private final TextField valueField = new TextField("0.0");

    public ManualStaticUiStrategy() {
        container.getChildren().addAll(
                new Label("Valor Fijo Constant:"),
                valueField,
                new Label("Este valor será constante en el tiempo."));
    }

    @Override
    public String getDisplayName() {
        return "Manual: Valor Estático";
    }

    @Override
    public String getStrategyCode() {
        return "MANUAL_STATIC";
    }

    @Override
    public SensorStrategyCategory getCategory() {
        return SensorStrategyCategory.MANUAL;
    }

    @Override
    public Node render() {
        return container;
    }

    @Override
    public boolean validate() {
        try {
            Double.parseDouble(valueField.getText());
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    @Override
    public Map<String, Object> extractConfiguration() {
        return Map.of("value", Double.parseDouble(valueField.getText()));
    }

    @Override
    public void reset() {
        valueField.setText("0.0");
    }

    @Override
    public void populate(Map<String, Object> config) {
        if (config != null && config.containsKey("value")) {
            valueField.setText(String.valueOf(config.get("value")));
        }
    }
}
