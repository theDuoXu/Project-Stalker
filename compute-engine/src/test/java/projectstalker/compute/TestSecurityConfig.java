package projectstalker.compute;

import org.springframework.boot.test.context.TestConfiguration;
import org.springframework.context.annotation.Bean;
import org.springframework.security.oauth2.jwt.JwtDecoder;

import static org.mockito.Mockito.mock;

import org.springframework.context.annotation.Primary;

@TestConfiguration
public class TestSecurityConfig {

    @Bean
    public JwtDecoder jwtDecoder() {
        return mock(JwtDecoder.class);
    }

    @Bean
    @Primary
    public org.springframework.data.redis.core.RedisTemplate<String, Object> redisTemplate() {
        return mock(org.springframework.data.redis.core.RedisTemplate.class);
    }

    @Bean
    @Primary
    public org.springframework.data.redis.connection.RedisConnectionFactory redisConnectionFactory() {
        return mock(org.springframework.data.redis.connection.RedisConnectionFactory.class);
    }
}
