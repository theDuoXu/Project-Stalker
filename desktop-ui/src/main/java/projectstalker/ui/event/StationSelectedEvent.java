package projectstalker.ui.event;

import org.springframework.context.ApplicationEvent;

public class StationSelectedEvent extends ApplicationEvent {

    private final String stationId;

    public StationSelectedEvent(Object source, String stationId) {
        super(source);
        this.stationId = stationId;
    }

    public String getStationId() {
        return stationId;
    }
}
