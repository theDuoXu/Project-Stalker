package projectstalker.ui.view.components.strategies;

import javafx.scene.Node;
import java.util.Map;

public interface SensorUiStrategy {
    String getDisplayName(); // Ej: "Virtual: Ondulatoria"

    String getStrategyCode(); // Ej: "VIRTUAL_SINE"

    SensorStrategyCategory getCategory(); // Clasificación (MANUAL, REAL, VIRTUAL)

    Node render(); // Devuelve el panel JavaFX

    boolean validate(); // Valida sus propios campos

    Map<String, Object> extractConfiguration(); // Devuelve el map para el JSONB

    void reset(); // Limpia campos

    default void populate(Map<String, Object> config) {
    } // Rellena campos para edición
}