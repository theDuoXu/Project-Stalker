package projectstalker.ui.view.components.strategies;

import javafx.scene.Node;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.VBox;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.Map;

@Component
public class VirtualRandomUiStrategy implements SensorUiStrategy {

    private final VBox container = new VBox(10);
    private final TextField minField = new TextField("0.0");
    private final TextField maxField = new TextField("10.0");

    public VirtualRandomUiStrategy() {
        GridPane grid = new GridPane();
        grid.setHgap(10);
        grid.setVgap(10);
        grid.addRow(0, new Label("Valor Mínimo:"), minField);
        grid.addRow(1, new Label("Valor Máximo:"), maxField);

        container.getChildren().addAll(new Label("Generador Aleatorio (Ruido Blanco)"), grid);
    }

    @Override
    public String getDisplayName() {
        return "Virtual: Aleatorio (Ruido)";
    }

    @Override
    public String getStrategyCode() {
        return "VIRTUAL_RANDOM";
    }

    @Override
    public SensorStrategyCategory getCategory() {
        return SensorStrategyCategory.VIRTUAL;
    }

    @Override
    public Node render() {
        return container;
    }

    @Override
    public boolean validate() {
        try {
            double min = Double.parseDouble(minField.getText());
            double max = Double.parseDouble(maxField.getText());
            return min <= max;
        } catch (Exception e) {
            return false;
        }
    }

    @Override
    public Map<String, Object> extractConfiguration() {
        Map<String, Object> map = new HashMap<>();
        map.put("min", Double.parseDouble(minField.getText()));
        map.put("max", Double.parseDouble(maxField.getText()));
        return map;
    }

    @Override
    public void reset() {
        minField.setText("0.0");
        maxField.setText("10.0");
    }
}
