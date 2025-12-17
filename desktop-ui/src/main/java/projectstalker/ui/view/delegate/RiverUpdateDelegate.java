package projectstalker.ui.view.delegate;

import javafx.beans.value.ObservableValue;
import javafx.scene.canvas.Canvas;
import javafx.scene.control.Tab;

/**
 * DELEGADO DE ACTUALIZACIÓN:
 * Centraliza la lógica de "Si cambia X, redibuja Y o marca la pestaña Z como sucia".
 */
public class RiverUpdateDelegate {

    private final Tab morphologyTab;
    private final Tab noiseTab;
    private final Tab hydrologyTab;

    // Acciones de redibujado (pasadas desde el Controller)
    private final Runnable redrawMorphology;
    private final Runnable redrawNoise;
    private final Runnable redrawHydrology;

    public RiverUpdateDelegate(Tab morphologyTab, Tab noiseTab, Tab hydrologyTab,
                               Runnable redrawMorphology, Runnable redrawNoise, Runnable redrawHydrology) {
        this.morphologyTab = morphologyTab;
        this.noiseTab = noiseTab;
        this.hydrologyTab = hydrologyTab;
        this.redrawMorphology = redrawMorphology;
        this.redrawNoise = redrawNoise;
        this.redrawHydrology = redrawHydrology;
    }

    /**
     * Grupo 1: Cambios que afectan la GEOMETRÍA (Morfología)
     */
    @SafeVarargs
    public final void trackMorphologyChanges(ObservableValue<? extends Number>... properties) {
        for (var prop : properties) {
            prop.addListener((obs, old, val) -> {
                if (morphologyTab.isSelected()) {
                    redrawMorphology.run();
                } else {
                    markTabDirty(morphologyTab);
                    // Si la geometría cambia, la hidrología también queda obsoleta
                    markTabDirty(hydrologyTab);
                }
            });
        }
    }

    /**
     * Grupo 2: Cambios que afectan RUIDO + GEOMETRÍA + HIDROLOGÍA (Shared)
     */
    @SafeVarargs
    public final void trackSharedChanges(ObservableValue<? extends Number>... properties) {
        for (var prop : properties) {
            prop.addListener((obs, old, val) -> {
                if (morphologyTab.isSelected()) redrawMorphology.run();
                else if (noiseTab.isSelected()) redrawNoise.run();
                else {
                    markTabDirty(morphologyTab);
                    markTabDirty(noiseTab);
                }
                markTabDirty(hydrologyTab); // También afecta hidrología
            });
        }
    }

    /**
     * Grupo 3: Cambios que afectan solo a la HIDROLOGÍA (Física/Química)
     */
    @SafeVarargs
    public final void trackHydrologyChanges(ObservableValue<? extends Number>... properties) {
        for (var prop : properties) {
            prop.addListener((obs, old, val) -> {
                if (hydrologyTab.isSelected()) {
                    redrawHydrology.run();
                } else {
                    markTabDirty(hydrologyTab);
                }
            });
        }
    }

    public final void trackCanvasSizeChanges(Canvas canvas, Runnable redrawStrategy){
        canvas.widthProperty().addListener((e) -> redrawStrategy.run());
        canvas.heightProperty().addListener((e) -> redrawStrategy.run());
    }

    // --- Gestión Visual de Tabs ---

    public void markTabDirty(Tab tab) {
        if (tab.isSelected()) return; // No marcar si estamos en la pestaña
        if (!tab.getText().contains("✱")) {
            tab.setText(tab.getText() + " ✱");
        }
        tab.setStyle("-fx-background-color: #FFA50040; -fx-text-fill: -color-accent-fg;");
    }

    public void cleanTab(Tab tab) {
        tab.setStyle(null);
        tab.setText(tab.getText().replace(" ✱", ""));
    }
}