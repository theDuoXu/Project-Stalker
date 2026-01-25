package projectstalker.ui.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import projectstalker.domain.dto.rule.RuleConfigDTO;
import projectstalker.ui.config.ApiClientConfig;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

@Service
@Slf4j
@RequiredArgsConstructor
public class RuleClientService {

    private final WebClient apiClient;

    private WebClient getClient() {
        return apiClient;
    }

    public Flux<RuleConfigDTO> getAllRules() {
        return getClient().get()
                .uri("/api/rules")
                .retrieve()
                .bodyToFlux(RuleConfigDTO.class);
    }

    public Mono<RuleConfigDTO> saveRule(RuleConfigDTO rule) {
        return getClient().post()
                .uri("/api/rules")
                .bodyValue(rule)
                .retrieve()
                .bodyToMono(RuleConfigDTO.class);
    }
}
