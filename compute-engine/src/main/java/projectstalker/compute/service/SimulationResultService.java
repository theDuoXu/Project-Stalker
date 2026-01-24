package projectstalker.compute.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;
import projectstalker.domain.simulation.IManningResult;

import java.time.Duration;

@Service
@Slf4j
@RequiredArgsConstructor
public class SimulationResultService {

    private final RedisTemplate<String, Object> redisTemplate;
    // Note: Storing complex Java objects (IManningResult) in Redis requires proper
    // serializer or custom binary format.
    // For MVP/Demo, using default JDK serialization or JSON if compatible (requires
    // IManningResult to be POJO).
    // If IManningResult contains large float arrays, binary is better.
    // We will assume RedisTemplate is configured for Object/Byte[] or similar.

    private static final String KEY_PREFIX = "sim_result:";

    public void saveResult(String simId, IManningResult result) {
        String key = KEY_PREFIX + simId;
        try {
            redisTemplate.opsForValue().set(key, result, Duration.ofHours(24));
            log.info("Saved simulation result to Redis: {}", simId);
        } catch (Exception e) {
            log.error("Failed to save result to Redis for {}", simId, e);
        }
    }

    public IManningResult getResult(String simId) {
        String key = KEY_PREFIX + simId;
        try {
            // Casting might fail if serializer mismatch, assume configured correctly in
            // RedisConfig default
            return (IManningResult) redisTemplate.opsForValue().get(key);
        } catch (Exception e) {
            log.error("Failed to get result from Redis for {}", simId, e);
            return null;
        }
    }
}
