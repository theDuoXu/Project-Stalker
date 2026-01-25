package projectstalker.ui.event;

import org.springframework.context.ApplicationEvent;

public class SensorListRefreshEvent extends ApplicationEvent {
    public SensorListRefreshEvent(Object source) {
        super(source);
    }
}
