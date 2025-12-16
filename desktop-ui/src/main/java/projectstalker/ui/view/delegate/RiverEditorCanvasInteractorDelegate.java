package projectstalker.ui.view.delegate;

import javafx.application.Platform;
import javafx.scene.canvas.Canvas;
import org.springframework.stereotype.Component;

/**
 * Gestiona la interacción del usuario con los Canvas (Mouse tracking)
 * y la detección de cambios de tema/escena para redibujado.
 */
@Component
public class RiverEditorCanvasInteractorDelegate {

    private double mouseX = -1;
    private double mouseY = -1;

    /**
     * Conecta los listeners a uno o varios canvas.
     *
     * @param onRepaint     Acción a ejecutar cuando el mouse se mueve o sale.
     * @param onThemeReload Acción a ejecutar cuando la escena cambia (CSS reload).
     * @param canvases      Lista de canvas a monitorear.
     */
    public void bind(Runnable onRepaint, Runnable onThemeReload, Canvas... canvases) {
        for (Canvas canvas : canvases) {
            setupMouseListeners(canvas, onRepaint);
            setupSceneListeners(canvas, onRepaint, onThemeReload);
        }
    }

    private void setupMouseListeners(Canvas canvas, Runnable onRepaint) {
        canvas.setOnMouseMoved(e -> {
            this.mouseX = e.getX();
            this.mouseY = e.getY();
            onRepaint.run();
        });

        canvas.setOnMouseExited(e -> {
            this.mouseX = -1;
            this.mouseY = -1;
            onRepaint.run();
        });
    }

    private void setupSceneListeners(Canvas canvas, Runnable onRepaint, Runnable onThemeReload) {
        // Detectar cuando el Canvas se añade a una escena para cargar CSS/Theme
        canvas.sceneProperty().addListener((obs, oldVal, newVal) -> {
            if (newVal != null) {
                Platform.runLater(() -> {
                    if (onThemeReload != null) onThemeReload.run();
                    onRepaint.run();
                });
            }
        });
    }

    public double getMouseX() {
        return mouseX;
    }

    public double getMouseY() {
        return mouseY;
    }
}