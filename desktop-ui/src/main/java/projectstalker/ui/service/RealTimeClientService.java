package projectstalker.ui.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.messaging.converter.MappingJackson2MessageConverter;
import org.springframework.messaging.simp.stomp.*;
import org.springframework.stereotype.Service;
import org.springframework.web.socket.client.WebSocketClient;
import org.springframework.web.socket.client.standard.StandardWebSocketClient;
import org.springframework.web.socket.messaging.WebSocketStompClient;
import projectstalker.domain.simulation.SimulationResponseDTO;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Sinks;

import java.lang.reflect.Type;
import java.util.concurrent.ExecutionException;

@Slf4j
@Service
public class RealTimeClientService {

    private StompSession session;
    private final Sinks.Many<String> connectionStatusSink = Sinks.many().multicast().onBackpressureBuffer();

    // URL del servidor (Hardcoded por ahora, idealmente en properties)
    private static final String WS_URL = "ws://localhost:8080/ws-stalker";

    public void connect() {
        WebSocketClient client = new StandardWebSocketClient();
        WebSocketStompClient stompClient = new WebSocketStompClient(client);
        stompClient.setMessageConverter(new MappingJackson2MessageConverter());

        log.info("Conectando a WebSocket: {}", WS_URL);

        stompClient.connectAsync(WS_URL, new StompSessionHandlerAdapter() {
            @Override
            public void afterConnected(StompSession session, StompHeaders connectedHeaders) {
                log.info("WebSocket CONECTADO. Session ID: {}", session.getSessionId());
                RealTimeClientService.this.session = session;
                connectionStatusSink.tryEmitNext("CONNECTED");
            }

            @Override
            public void handleException(StompSession session, StompCommand command, StompHeaders headers,
                    byte[] payload, Throwable exception) {
                log.error("Error STOMP", exception);
                connectionStatusSink.tryEmitNext("ERROR");
            }

            @Override
            public void handleTransportError(StompSession session, Throwable exception) {
                log.error("Error Transporte WS", exception);
                connectionStatusSink.tryEmitNext("DISCONNECTED");
            }
        });
    }

    public Flux<String> getConnectionStatus() {
        return connectionStatusSink.asFlux();
    }

    /**
     * Se suscribe a los eventos de una simulación específica.
     * Topic: /topic/simulation/{simId}
     */
    public Flux<SimulationResponseDTO> subscribeToSimulation(String simId) {
        if (session == null || !session.isConnected()) {
            return Flux.error(new IllegalStateException("No hay conexión WebSocket activa."));
        }

        return Flux.create(sink -> {
            String topic = "/topic/simulation/" + simId;
            log.debug("Suscribiéndose a: {}", topic);

            StompSession.Subscription sub = session.subscribe(topic, new StompFrameHandler() {
                @Override
                public Type getPayloadType(StompHeaders headers) {
                    return SimulationResponseDTO.class;
                }

                @Override
                public void handleFrame(StompHeaders headers, Object payload) {
                    if (payload instanceof SimulationResponseDTO dto) {
                        sink.next(dto);
                    }
                }
            });

            // Al cancelar el Flux, nos desuscribimos del topic
            sink.onDispose(sub::unsubscribe);
        });
    }
}
