package projectstalker.ui.view.components;

import javafx.fxml.FXML;
import javafx.scene.control.Label;
import org.kordamp.ikonli.javafx.FontIcon;
import projectstalker.domain.dto.sensor.SensorResponseDTO;

public class SensorCardController {

    @FXML private Label nameLabel;
    @FXML private Label valueLabel;
    @FXML private Label locationLabel;
    @FXML private FontIcon typeIcon;
    @FXML private FontIcon statusIcon;

    public void setData(SensorResponseDTO data) {
        nameLabel.setText(data.name());

        // Formatear valor (si hay lecturas, coger la última)
        if (data.values() != null && !data.values().isEmpty()) {
            double val = data.values().get(data.values().size() - 1).value();
            valueLabel.setText(String.format("%.2f %s", val, data.unit()));
        } else {
            valueLabel.setText("-- " + data.unit());
        }

        // Iconografía dinámica
        switch (data.signalType()) {
            case "PH" -> typeIcon.setIconLiteral("mdi2t-test-tube");
            case "TEMPERATURE" -> typeIcon.setIconLiteral("mdi2t-thermometer");
            case "FLOW" -> typeIcon.setIconLiteral("mdi2w-water");
            default -> typeIcon.setIconLiteral("mdi2s-sensors");
        }
    }
}