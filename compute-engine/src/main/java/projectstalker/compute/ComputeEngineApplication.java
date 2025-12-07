package projectstalker.compute;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.ComponentScan;

// @SpringBootApplication le dice a Spring: "Aquí empieza todo".
// Escanea automáticamente componentes en este paquete y sub-paquetes.
// Usamos scanBasePackages para asegurarnos de que pilla tus clases de física si están en 'projectstalker.physics'
@SpringBootApplication(scanBasePackages = "projectstalker")
public class ComputeEngineApplication {

    public static void main(String[] args) {
        // Carga la librería nativa al arrancar la JVM del servidor
        // (Esto es opcional si lo haces con -Djava.library.path, pero es una buena práctica defensiva)
        try {
            System.loadLibrary("manning_solver");
            System.out.println(">>> Librería Nativa JNI cargada correctamente en el arranque.");
        } catch (UnsatisfiedLinkError e) {
            System.err.println(">>> ADVERTENCIA: No se pudo cargar la librería nativa automáticamente. " +
                    "Asegúrate de pasar -Djava.library.path o que esté en el sistema.");
        }

        SpringApplication.run(ComputeEngineApplication.class, args);
    }
}