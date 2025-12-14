package projectstalker.ui;

import atlantafx.base.theme.NordDark;
import javafx.application.Application;
import javafx.application.Platform;
import javafx.stage.Stage;
import org.springframework.boot.builder.SpringApplicationBuilder;
import org.springframework.context.ConfigurableApplicationContext;
import projectstalker.ui.view.StageInitializer;

public class StalkerFxApp extends Application {

    private ConfigurableApplicationContext applicationContext;

    @Override
    public void init() {
        // Inicializamos Spring Boot, pasándole esta clase y los argumentos
        applicationContext = new SpringApplicationBuilder(StalkerUiLauncher.class)
                .run(getParameters().getRaw().toArray(new String[0]));
    }

    @Override
    public void start(Stage stage) {
        // Aplicar tema profesional (AtlantaFX Nord Dark) (Aquí es application de JavaFX no confundir con Application context!)
        Application.setUserAgentStylesheet(new NordDark().getUserAgentStylesheet());

        // Publicar evento para que Spring maneje la Stage (StageInitializer escuchará esto)
        applicationContext.publishEvent(new StageReadyEvent(stage));
    }

    @Override
    public void stop() {
        // Cerrar contexto de Spring limpiamente al cerrar la ventana
        applicationContext.close();
        Platform.exit();
    }
}