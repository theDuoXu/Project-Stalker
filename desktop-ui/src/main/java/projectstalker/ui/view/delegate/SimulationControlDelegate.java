package projectstalker.ui.view.delegate;

import javafx.application.Platform;
import javafx.scene.control.Label;
import org.springframework.context.ApplicationEventPublisher;
import projectstalker.ui.event.TransitoryStatusUpdateEvent;
import projectstalker.ui.service.SimulationEngine;
import projectstalker.ui.viewmodel.StatusType;
import projectstalker.ui.viewmodel.StatusViewModel;

public class SimulationControlDelegate {

    private final SimulationEngine simEngine;
    private final ApplicationEventPublisher eventPublisher;
    private final Label timeLabel;
    private final Label speedLabel;

    public SimulationControlDelegate(SimulationEngine simEngine,
                                     ApplicationEventPublisher eventPublisher,
                                     Label timeLabel,
                                     Label speedLabel) {
        this.simEngine = simEngine;
        this.eventPublisher = eventPublisher;
        this.timeLabel = timeLabel;
        this.speedLabel = speedLabel;
    }

    public void restart() {
        notify("Reiniciando...", StatusType.DEFAULT);
        simEngine.restartTime();
        updateSpeedLabel();
    }

    public void rewind() {
        notify("Rebobinando...", StatusType.DEFAULT);
        simEngine.setPlaybackSpeed(-2.0);
        updateSpeedLabel();
    }

    public void pause() {
        notify("Simulación pausada", StatusType.WARNING);
        simEngine.setPlaybackSpeed(0.0);
        updateSpeedLabel();
    }

    public void play() {
        notify("Simulación iniciada", StatusType.DEFAULT);
        simEngine.setPlaybackSpeed(1.0);
        updateSpeedLabel();
    }

    public void accelerate() {
        notify("Acelerando simulación", StatusType.DEFAULT);
        double current = simEngine.getPlaybackSpeed();
        if (current <= 0) {
            simEngine.setPlaybackSpeed(2.0);
        } else {
            simEngine.setPlaybackSpeed(Math.min(64.0, current * 2.0));
        }
        updateSpeedLabel();
    }

    /**
     * Actualiza la etiqueta de tiempo. Seguro para llamar desde hilos de fondo.
     */
    public void updateTime(double totalSeconds) {
        long totalSecs = (long) totalSeconds;
        long days = totalSecs / 86400;
        long remainder = totalSecs % 86400;
        long hours = remainder / 3600;
        long minutes = (remainder % 3600) / 60;

        String formattedTime = String.format("T+ %02dd %02d:%02d", days, hours, minutes);

        Platform.runLater(() -> timeLabel.setText(formattedTime));
    }

    private void updateSpeedLabel() {
        double speed = simEngine.getPlaybackSpeed();

        // 1. Limpieza: Eliminamos cualquier clase de estado previa
        speedLabel.getStyleClass().removeAll("status-success", "status-warning", "status-error", "status-muted");

        if (speed == 0) {
            speedLabel.setText("PAUSED");
            speedLabel.getStyleClass().add("status-muted");
        } else {
            speedLabel.setText(String.format("%.1fx", speed));

            if (speed > 0) {
                // Velocidad positiva -> Verde
                speedLabel.getStyleClass().add("status-success");
            } else {
                // Velocidad negativa -> Amarillo
                speedLabel.getStyleClass().add("status-warning");
            }
        }
    }

    private void notify(String msg, StatusType type) {
        StatusType finalType = (type == null) ? StatusType.SUCCESS : type;
        eventPublisher.publishEvent(
                new TransitoryStatusUpdateEvent(msg, StatusViewModel.TransitionTime.SHORT, finalType));
    }
}