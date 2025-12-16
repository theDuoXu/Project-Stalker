package projectstalker.ui.viewmodel;

import javafx.animation.PauseTransition;
import javafx.application.Platform;
import javafx.beans.property.ObjectProperty;
import javafx.beans.property.SimpleObjectProperty;
import javafx.beans.property.SimpleStringProperty;
import javafx.beans.property.StringProperty;
import javafx.util.Duration;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Component;
import projectstalker.ui.event.PermanentStatusUpdateEvent;
import projectstalker.ui.event.TransitoryStatusUpdateEvent;

import java.util.LinkedList;
import java.util.Queue;

public class StatusViewModel {

    private final StringProperty statusMessage = new SimpleStringProperty();
    // Propiedad para el tipo
    private final ObjectProperty<StatusType> statusType = new SimpleObjectProperty<>(StatusType.DEFAULT);

    private final StatusTarget target;

    // Estado permanente de respaldo
    private String permanentMessage = "";
    private StatusType permanentType = StatusType.DEFAULT;

    private final Queue<TransitoryStatusUpdateEvent> notificationQueue = new LinkedList<>();
    private boolean isShowingTransitory = false;
    private PauseTransition currentTransition;

    public StatusViewModel(StatusTarget target) {
        this.target = target;
        // Inicialización por defecto según el target
        if (target == StatusTarget.APP) {
            this.permanentMessage = "Sistema DSS Inicializado.";
        }
    }

    // --- Getters de Propiedades para JavaFX ---
    public StringProperty statusMessageProperty() { return statusMessage; }
    public ObjectProperty<StatusType> statusTypeProperty() { return statusType; }

    // --- Lógica ---

    @EventListener
    public void handlePermanentStatusUpdate(PermanentStatusUpdateEvent event) {
        if (event.target() != this.target) return;

        Platform.runLater(() -> {
            // REGLA DE PRIORIDAD: Un cambio de estado permanente (ej: Desconexión)
            // mata cualquier notificación pendiente. La realidad manda.
            stopCurrentTransition();
            notificationQueue.clear();
            isShowingTransitory = false;

            // Guardamos y mostramos la nueva realidad
            this.permanentMessage = event.message();
            this.permanentType = event.type();

            updateState(this.permanentMessage, this.permanentType);
        });
    }

    @EventListener
    public void handleTransitoryStatusUpdate(TransitoryStatusUpdateEvent event) {
        if (event.target() != this.target) return;

        // Si es permanente disfrazado, lo delegamos
        if (event.transitionTime() == TransitionTime.PERMANENT) {
            handlePermanentStatusUpdate(event.asPermanentEvent());
            return;
        }

        Platform.runLater(() -> {
            // 1. Añadimos a la cola
            notificationQueue.add(event);

            // 2. Si no estamos mostrando nada, arrancamos la cola
            if (!isShowingTransitory) {
                processNextInQueue();
            }
        });
    }

    private void processNextInQueue() {
        // Si la cola está vacía, volvemos al caso base (Rompe recursividad)
        if (notificationQueue.isEmpty()) {
            isShowingTransitory = false;
            updateState(this.permanentMessage, this.permanentType);
            return;
        }

        // Sacamos el siguiente evento
        TransitoryStatusUpdateEvent nextEvent = notificationQueue.poll();
        isShowingTransitory = true;

        // Actualizamos UI
        updateState(nextEvent.message(), nextEvent.type());

        // Programamos el timer
        stopCurrentTransition();
        currentTransition = new PauseTransition(nextEvent.transitionTime().toJavaFXDuration());

        // RECURSIVIDAD: Al terminar, llamamos a este mismo método
        currentTransition.setOnFinished(e -> processNextInQueue());

        currentTransition.play();
    }

    private void stopCurrentTransition() {
        if (currentTransition != null) {
            currentTransition.stop();
            currentTransition = null;
        }
    }

    private void updateState(String message, StatusType type) {
        this.statusMessage.set(message);
        this.statusType.set(type);
    }

    public enum TransitionTime {
        /**
         * El mensaje de estado es permanente y nunca desaparece automáticamente.
         * Requiere una actualización explícita
         */
        PERMANENT(0),

        /**
         * El mensaje desaparece después de un retardo muy corto (1 segundo).
         * Ideal para feedback instantáneo
         */
        IMMEDIATE(1000), // 1.0 segundo

        /**
         * El mensaje desaparece después de un corto período de tiempo (3 segundos).
         * Ideal para notificaciones breves y no críticas (ej: "Guardado con éxito").
         */
        SHORT(3000), // 3.0 segundos

        /**
         * El mensaje desaparece después de un período de tiempo moderado (6 segundos).
         * Útil para notificaciones importantes que requieren la atención del usuario (ej: "Conexión restaurada").
         */
        MEDIUM(6000); // 6.0 segundos

        // Campo para almacenar la duración en milisegundos
        private final long durationMillis;

        /**
         * Constructor privado para asignar la duración a cada constante del enum.
         * @param durationMillis La duración en milisegundos.
         */
        TransitionTime(long durationMillis) {
            this.durationMillis = durationMillis;
        }

        /**
         * Convierte la duración a un objeto Duration de JavaFX.
         * Esto simplifica su uso con PauseTransition y otros componentes de JavaFX.
         * @return Un objeto javafx.util.Duration.
         */
        public Duration toJavaFXDuration() {
            if (this.durationMillis <= 0){ // Caso permanente
                return Duration.INDEFINITE;
            }
            return Duration.millis(durationMillis);
        }
    }
}