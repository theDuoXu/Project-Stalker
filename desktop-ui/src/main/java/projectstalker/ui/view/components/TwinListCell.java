package projectstalker.ui.view.components;

import javafx.geometry.Pos;
import javafx.scene.control.Label;
import javafx.scene.control.ListCell;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.layout.Region;
import javafx.scene.layout.Priority;
import org.kordamp.ikonli.javafx.FontIcon;
import projectstalker.domain.dto.twin.TwinSummaryDTO;

public class TwinListCell extends ListCell<TwinSummaryDTO> {

    private final HBox root;
    private final Label nameLabel;
    private final Label descLabel;
    private final Label metricsLabel;
    private final FontIcon statusIcon;

    public TwinListCell() {
        // 1. Icono de Estado (Izquierda)
        statusIcon = new FontIcon("mdi2w-water");
        statusIcon.setIconSize(24);

        // 2. Información Principal (Nombre y Descripción)
        nameLabel = new Label();
        nameLabel.setStyle("-fx-font-weight: bold; -fx-font-size: 13px;");

        descLabel = new Label();
        descLabel.setStyle("-fx-text-fill: -color-fg-muted; -fx-font-size: 11px;");

        VBox infoBox = new VBox(2, nameLabel, descLabel);
        infoBox.setAlignment(Pos.CENTER_LEFT);

        // 3. Métricas (Derecha - Longitud y Celdas)
        metricsLabel = new Label();
        metricsLabel.setStyle("-fx-font-size: 10px; -fx-text-fill: -color-accent-fg;");

        // Espaciador para empujar métricas a la derecha
        Region spacer = new Region();
        HBox.setHgrow(spacer, Priority.ALWAYS);

        // 4. Layout Final
        root = new HBox(10, statusIcon, infoBox, spacer, metricsLabel);
        root.setAlignment(Pos.CENTER_LEFT);
        root.setPadding(new javafx.geometry.Insets(8));
    }

    @Override
    protected void updateItem(TwinSummaryDTO item, boolean empty) {
        super.updateItem(item, empty);

        if (empty || item == null) {
            setText(null);
            setGraphic(null);
        } else {
            // Lógica de presentación
            nameLabel.setText(item.name());
            descLabel.setText(item.description());

            // Formateo de métricas
            metricsLabel.setText(String.format("%.1f km | %d celdas",
                    item.totalLengthKm(), item.cellCount()));

            // Lógica visual simple para estado (stub)
            if (item.cellCount() > 0) {
                statusIcon.setIconColor(javafx.scene.paint.Color.web("#81A1C1")); // Nord Blue
            } else {
                statusIcon.setIconColor(javafx.scene.paint.Color.web("#BF616A")); // Nord Red
            }

            setText(null);
            setGraphic(root);
        }
    }
}