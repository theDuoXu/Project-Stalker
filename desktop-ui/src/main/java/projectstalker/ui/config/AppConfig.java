package projectstalker.ui.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.PropertySource;
import org.springframework.context.support.PropertySourcesPlaceholderConfigurer;
import projectstalker.ui.viewmodel.StatusTarget;
import projectstalker.ui.viewmodel.StatusViewModel;

@Configuration
@PropertySource("classpath:application.properties")
public class AppConfig {

    /**
     * Este Bean estático es OBLIGATORIO para que @Value funcione
     * cuando se carga un PropertySource manualmente en algunas versiones de Spring.
     * Actúa como el puente entre el archivo y la inyección de dependencias.
     */
    @Bean
    public static PropertySourcesPlaceholderConfigurer propertySourcesPlaceholderConfigurer() {
        return new PropertySourcesPlaceholderConfigurer();
    }

    @Bean
    public StatusViewModel mainStatusViewModel() {
        return new StatusViewModel(StatusTarget.APP);
    }

    @Bean
    public StatusViewModel hpcStatusViewModel() {
        return new StatusViewModel(StatusTarget.HPC);
    }
}