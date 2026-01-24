package projectstalker.compute.api;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.messaging.handler.annotation.DestinationVariable;
import org.springframework.messaging.handler.annotation.MessageMapping;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.stereotype.Controller;

@Controller
@Slf4j
@RequiredArgsConstructor
public class SimSocketController {

    private final SimpMessagingTemplate messagingTemplate;

    /**
     * Broadcasts simulation progress to subscribers of /topic/simulation/{simId}
     */
    public void sendProgressUpdate(String simId, Object progressData) {
        String destination = "/topic/simulation/" + simId;
        messagingTemplate.convertAndSend(destination, progressData);
        log.debug("Sent progress update to {}: {}", destination, progressData);
    }

    // Example endpoint if clients need to send commands via WS
    @MessageMapping("/simulation/{simId}/pause")
    public void pauseSimulation(@DestinationVariable String simId) {
        log.info("Client requested pause for simulation {}", simId);
        // Dispatch pause logic
    }
}
