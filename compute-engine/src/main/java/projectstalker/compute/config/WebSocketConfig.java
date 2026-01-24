package projectstalker.compute.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.messaging.simp.config.MessageBrokerRegistry;
import org.springframework.web.socket.config.annotation.EnableWebSocketMessageBroker;
import org.springframework.web.socket.config.annotation.StompEndpointRegistry;
import org.springframework.web.socket.config.annotation.WebSocketMessageBrokerConfigurer;

@Configuration
@EnableWebSocketMessageBroker
public class WebSocketConfig implements WebSocketMessageBrokerConfigurer {

    @Override
    public void configureMessageBroker(MessageBrokerRegistry config) {
        // Enable a simple in-memory broker for the given prefixes
        config.enableSimpleBroker("/topic", "/queue");
        // Application destination prefix for client sending messages
        config.setApplicationDestinationPrefixes("/app");
    }

    @Override
    public void registerStompEndpoints(StompEndpointRegistry registry) {
        // Registers the "/ws" endpoint, enabling SockJS fallback options so that
        // alternate transports can be used if WebSocket is not available.
        registry.addEndpoint("/ws")
                .setAllowedOriginPatterns("*") // Be careful in production
                .withSockJS();
    }
}
