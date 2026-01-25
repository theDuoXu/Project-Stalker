package projectstalker.ui.view.components.strategies;

import javafx.scene.Node;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.control.Tooltip;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.VBox;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.Map;

@Component
public class VirtualSineUiStrategy implements SensorUiStrategy {

    private final VBox container = new VBox(10);
    private final TextField offsetField = new TextField("7.0");
    private final TextField amplitudeField = new TextField("1.5");
    private final TextField freqField = new TextField("0.1");

    public VirtualSineUiStrategy() {
        GridPane grid = new GridPane();
        grid.setHgap(10);
        grid.setVgap(10);

        // Construcción de la UI programática
        grid.addRow(0, new Label("Offset (Valor Medio):"), offsetField);
        grid.addRow(1, new Label("Amplitud (Oscilación):"), amplitudeField);
        grid.addRow(2, new Label("Frecuencia (Hz):"), freqField);

        offsetField.setTooltip(new Tooltip("Valor central alrededor del cual oscila la señal"));

        container.getChildren().addAll(new Label("Parámetros de Onda Senoidal"), grid);
    }

    @Override
    public String getDisplayName() {
        return "Virtual: Ondulatoria (Seno)";
    }

    @Override
    public String getStrategyCode() {
        return "VIRTUAL_SINE";
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
        return isNum(offsetField) && isNum(amplitudeField) && isNum(freqField);
    }

    @Override
    public Map<String, Object> extractConfiguration() {
        Map<String, Object> map = new HashMap<>();
        map.put("offset", Double.parseDouble(offsetField.getText()));
        map.put("amplitude", Double.parseDouble(amplitudeField.getText()));
        map.put("frequency", Double.parseDouble(freqField.getText()));
        return map;
    }

    @Override
    public void reset() {
        offsetField.setText("7.0");
        amplitudeField.setText("1.5");
        freqField.setText("0.1");
    }

    @Override
    public void populate(Map<String, Object> config) {
        if (config == null)
            return;
        if (config.containsKey("amplitude"))
            amplitudeField.setText(String.valueOf(((Number) config.get("amplitude")).doubleValue()));
        if (config.containsKey("frequency"))
            freqField.setText(String.valueOf(((Number) config.get("frequency")).doubleValue()));
        if (config.containsKey("offset"))
            offsetField.setText(String.valueOf(((Number) config.get("offset")).doubleValue()));
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