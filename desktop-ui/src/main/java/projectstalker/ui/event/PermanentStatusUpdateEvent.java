package projectstalker.ui.event;

import projectstalker.ui.viewmodel.StatusTarget;
import projectstalker.ui.viewmodel.StatusType;

public record PermanentStatusUpdateEvent(String message, StatusType type, StatusTarget target) {

    public PermanentStatusUpdateEvent(String message) {
        this(message, StatusType.DEFAULT, StatusTarget.APP);
    }

    public PermanentStatusUpdateEvent(String message, StatusType type) {
        this(message, type, StatusTarget.APP);
    }
}