package projectstalker.ui.viewmodel;

import javafx.animation.PauseTransition;
import javafx.application.Platform;
import javafx.beans.property.SimpleStringProperty;
import javafx.beans.property.StringProperty;
import javafx.util.Duration;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Component;
import projectstalker.ui.event.PermanentStatusUpdateEvent;
import projectstalker.ui.event.TransitoryStatusUpdateEvent;

@Component
public class StatusViewModel {

    private final StringProperty statusMessage = new SimpleStringProperty();

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

    // Último estado permanente
    private String permanentStatus = "Sistema DSS Inicializado.";

    public StringProperty statusMessageProperty() {
        return statusMessage;
    }

    public void setStatusMessage(String message) {
        this.statusMessage.set(message);
    }

    public String getStatusMessage() {
        return statusMessage.get();
    }

    @EventListener
    public void handlePermanentStatusUpdate(PermanentStatusUpdateEvent event) {
        // JavaFX requiere que las actualizaciones de UI (y sus propiedades) se realicen en el hilo de la aplicación.
        // Platform.runLater asegura esto.
        Platform.runLater(() -> {
            this.permanentStatus = event.message();
            setStatusMessage(event.message());
        });
    }

    @EventListener
    public void handleTransitoryStatusUpdate(TransitoryStatusUpdateEvent event) {

        // Evitar que un evento transitorio permanente anule el estado persistente.
        // Los eventos permanentes deben usar PermanentStatusUpdateEvent.
        if (event.transitionTime() == TransitionTime.PERMANENT) {
            handlePermanentStatusUpdate(event.asPermanentEvent());
            return;
        }

        Platform.runLater(() -> {
            this.statusMessage.set(event.message());
            PauseTransition pause = new PauseTransition(event.transitionTime().toJavaFXDuration());
            pause.setOnFinished(e -> {
                // Restaura el último mensaje permanente conocido
                this.statusMessage.set(this.permanentStatus);
            });
            pause.play();
        });
    }
}