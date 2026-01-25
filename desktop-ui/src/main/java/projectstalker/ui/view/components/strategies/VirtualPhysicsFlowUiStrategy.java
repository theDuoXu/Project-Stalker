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
public class VirtualPhysicsFlowUiStrategy implements SensorUiStrategy {

    private final VBox container = new VBox(10);
    private final TextField baseField = new TextField("25.0");
    private final TextField ampField = new TextField("5.0");
    private final TextField freqField = new TextField("0.02");

    public VirtualPhysicsFlowUiStrategy() {
        GridPane grid = new GridPane();
        grid.setHgap(10);
        grid.setVgap(10);

        grid.addRow(0, new Label("Caudal Base (m³/s):"), baseField);
        grid.addRow(1, new Label("Amplitud de Ruido:"), ampField);
        grid.addRow(2, new Label("Frecuencia (Suavidad):"), freqField);

        container.getChildren().addAll(new Label("Generador Físico (Perlin Noise)"), grid);
    }

    @Override
    public String getDisplayName() {
        return "Virtual: Perlin Flow Physics";
    }

    @Override
    public String getStrategyCode() {
        return "VIRTUAL_PHYSICS_FLOW";
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
        return isNum(baseField) && isNum(ampField) && isNum(freqField);
    }

    @Override
    public Map<String, Object> extractConfiguration() {
        Map<String, Object> map = new HashMap<>();
        // Maps to RandomFlowProfileGenerator constructor args
        map.put("baseDischarge", Double.parseDouble(baseField.getText()));
        map.put("noiseAmplitude", Double.parseDouble(ampField.getText()));
        map.put("frequency", Double.parseDouble(freqField.getText()));
        // Note: Generator takes float frequency, but we store as number here.
        return map;
    }

    @Override
    public void reset() {
        baseField.setText("25.0");
        ampField.setText("5.0");
        freqField.setText("0.02");
    }

    @Override
    public void populate(Map<String, Object> config) {
        if (config == null)
            return;
        if (config.containsKey("baseDischarge"))
            baseField.setText(String.valueOf(config.get("baseDischarge")));
        if (config.containsKey("noiseAmplitude"))
            ampField.setText(String.valueOf(config.get("noiseAmplitude")));
        if (config.containsKey("frequency"))
            freqField.setText(String.valueOf(config.get("frequency")));
    }

    private boolean isNum(TextField tf) {
        try {
            Double.parseDouble(tf.getText());
            return true;
        } catch (Exception e) {
            return false;
        }
    }
}
