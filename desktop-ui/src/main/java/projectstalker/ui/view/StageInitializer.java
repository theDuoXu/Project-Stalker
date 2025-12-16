package projectstalker.ui.view;

import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.ApplicationContext;
import org.springframework.context.ApplicationListener;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Component;
import projectstalker.ui.event.StageReadyEvent;

import java.io.IOException;
import java.net.URL;

@Component
@Slf4j
public class StageInitializer implements ApplicationListener<StageReadyEvent> {
    @Value("classpath:/fxml/main-view.fxml")
    private Resource mainViewResource;
    @Value("classpath:/css/styles.css")
    private Resource cssResource;

    private final String applicationTitle;
    private final ApplicationContext applicationContext;

    public StageInitializer(@Value("${spring.application.name:Project Stalker DSS}") String applicationTitle,
                            ApplicationContext applicationContext) {
        this.applicationTitle = applicationTitle;
        this.applicationContext = applicationContext;
    }

    @Override
    public void onApplicationEvent(StageReadyEvent event) {
        try {
            FXMLLoader fxmlLoader = new FXMLLoader(mainViewResource.getURL());
            fxmlLoader.setControllerFactory(applicationContext::getBean);
            Parent parent = fxmlLoader.load();

            Stage stage = event.getStage();
            stage.setTitle(applicationTitle);

            Scene scene = new Scene(parent, 1280, 800);

            if (cssResource.exists()) {
                scene.getStylesheets().add(cssResource.getURL().toExternalForm());
            } else {
                log.warn("WARN: Spring no pudo localizar classpath:/css/styles.css");
            }

            stage.setScene(scene);
            stage.show();

        } catch (IOException e) {
            throw new RuntimeException("Error fatal al cargar la UI", e);
        }
    }
}