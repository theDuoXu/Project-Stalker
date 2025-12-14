package projectstalker.ui.view;

import javafx.fxml.FXML;
import javafx.scene.control.Label;
import org.springframework.stereotype.Component;

@Component
public class MainController {

    @FXML
    public Label statusLabel;

    @FXML
    public void initialize() {
        // Esto se ejecuta justo después de cargar el FXML
        statusLabel.setText("Sistema DSS Inicializado. Motor HPC: Esperando conexión...");
    }

    // Aquí conectaremos luego los ViewModels
}