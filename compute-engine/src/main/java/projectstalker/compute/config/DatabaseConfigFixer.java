package projectstalker.compute.config;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.CommandLineRunner;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Component;

@Component
@RequiredArgsConstructor
@Slf4j
public class DatabaseConfigFixer implements CommandLineRunner {

    private final JdbcTemplate jdbcTemplate;

    @Override
    public void run(String... args) throws Exception {
        log.info("Checking and fixing database constraints...");
        try {
            // Drop the restrictive enum constraint created by Hibernate
            // This allows us to use new Enum values like RESOLVED
            jdbcTemplate.execute("ALTER TABLE alerts DROP CONSTRAINT IF EXISTS alerts_status_check");
            log.info("Dropped constraint 'alerts_status_check' successfully.");
        } catch (Exception e) {
            log.warn("Failed to drop constraint 'alerts_status_check'. It might not exist or other error: {}",
                    e.getMessage());
        }
    }
}
