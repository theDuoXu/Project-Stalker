package projectstalker.ui.view;

import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.ApplicationContext;
import org.springframework.context.ApplicationListener;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Component;
import projectstalker.ui.StageReadyEvent;

import java.io.IOException;

@Component
public class StageInitializer implements ApplicationListener<StageReadyEvent> {

    // Inyectamos el archivo FXML desde resources
    @Value("classpath:/fxml/main-view.fxml")
    private Resource mainViewResource;

    private final String applicationTitle;
    private final ApplicationContext applicationContext;

    // Inyección de dependencias vía constructor
    public StageInitializer(@Value("${spring.application.name:Project Stalker DSS}") String applicationTitle,
                            ApplicationContext applicationContext) {
        this.applicationTitle = applicationTitle;
        this.applicationContext = applicationContext;
    }

    @Override
    public void onApplicationEvent(StageReadyEvent event) {
        try {
            FXMLLoader fxmlLoader = new FXMLLoader(mainViewResource.getURL());

            // Obligamos a FXMLoader que no cree sus propias clases sino que use las gestionadas por Spring
            fxmlLoader.setControllerFactory(applicationContext::getBean);

            Parent parent = fxmlLoader.load();

            Stage stage = event.getStage();
            stage.setTitle(applicationTitle);

            // Escena inicial (1280x800 es un buen estándar HD)
            stage.setScene(new Scene(parent, 1280, 800));
            stage.show();

        } catch (IOException e) {
            throw new RuntimeException("Error fatal al cargar la UI", e);
        }
    }
}