package projectstalker.ui.event;

import projectstalker.ui.viewmodel.StatusTarget;
import projectstalker.ui.viewmodel.StatusType;
import projectstalker.ui.viewmodel.StatusViewModel;

public record TransitoryStatusUpdateEvent(
        String message,
        StatusType type, StatusViewModel.TransitionTime transitionTime,
        StatusTarget target
) {
    public TransitoryStatusUpdateEvent(String message, StatusViewModel.TransitionTime transitionTime) {
        this(message, StatusType.DEFAULT, transitionTime, StatusTarget.APP);
    }

    public TransitoryStatusUpdateEvent(String message, StatusViewModel.TransitionTime transitionTime, StatusType type) {
        this(message, type, transitionTime, StatusTarget.APP);
    }

    public PermanentStatusUpdateEvent asPermanentEvent(){
        if (transitionTime == StatusViewModel.TransitionTime.PERMANENT){
            return new PermanentStatusUpdateEvent(this.message);
        }
        throw new IllegalStateException(
                "No se puede convertir un evento transitorio con TransitionTime=" + transitionTime +
                        " a un evento permanente. Solo se permite si TransitionTime es PERMANENT."
        );
    }
}
