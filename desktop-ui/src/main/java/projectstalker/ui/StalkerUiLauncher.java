package projectstalker.ui;

import javafx.application.Application;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class StalkerUiLauncher {
    public static void main(String[] args) {
        Application.launch(StalkerFxApp.class, args);
    }
}