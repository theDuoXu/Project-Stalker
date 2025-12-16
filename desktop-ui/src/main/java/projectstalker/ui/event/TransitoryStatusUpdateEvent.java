package projectstalker.ui.event;

import projectstalker.ui.viewmodel.StatusViewModel;

public record TransitoryStatusUpdateEvent(String message, StatusViewModel.TransitionTime transitionTime){
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
