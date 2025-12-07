package projectstalker.compute;

import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import projectstalker.physics.jni.NativeManningGpuSingleton;

/**
 * Punto de entrada principal del Compute Engine (Backend).
 * <p>
 * Responsabilidades:
 * 1. Arrancar el contexto de Spring Boot (Web, Security, etc.).
 * 2. Verificar la integridad del puente JNI/CUDA antes de aceptar tráfico.
 */
@Slf4j
@SpringBootApplication(scanBasePackages = "projectstalker")
public class ComputeEngineApplication {

    public static void main(String[] args) {
        // El puerto se puede configurar vía args: --server.port=9090
        SpringApplication.run(ComputeEngineApplication.class, args);
    }

    /**
     * Bean de Verificación de Integridad ("Fail Fast").
     * Se ejecuta automáticamente justo después de que Spring levanta el contexto.
     * <p>
     * Intenta acceder al Singleton Nativo. Si la librería .so/.dll no se encuentra
     * o falla al cargar, este método captura el error y apaga el servidor inmediatamente.
     */
    @Bean
    public CommandLineRunner gpuIntegrityCheck() {
        return args -> {
            log.info(">>> BOOTSTRAP: Verificando disponibilidad del Motor Físico (JNI/CUDA)...");

            try {
                // 1. Carga de Librería (DLL/.so)
                NativeManningGpuSingleton instance = NativeManningGpuSingleton.getInstance();
                if (instance == null) throw new IllegalStateException("Singleton es null.");

                log.info(">>> [PASO 1/2] Librería nativa cargada.");

                // 2. Chequeo de Hardware Real
                int gpuCount = instance.getDeviceCount();

                if (gpuCount > 0) {
                    log.info(">>> [PASO 2/2] Hardware detectado: {} GPU(s) NVIDIA compatibles.", gpuCount);
                    log.info(">>> BOOTSTRAP: SISTEMA LISTO PARA CÓMPUTO.");
                } else {
                    log.error(">>> FATAL: La librería cargó, pero NO SE DETECTARON GPUS CUDA.");
                    log.error(">>> Posibles causas: Driver no instalado, GPU no compatible o en uso exclusivo.");
                    System.exit(1);
                }

            } catch (UnsatisfiedLinkError e) {
                log.error(">>> FATAL: No se encuentra la librería nativa.", e);
                System.exit(1);
            } catch (Exception e) {
                log.error(">>> FATAL: Error inesperado en arranque.", e);
                System.exit(1);
            }
        };
    }
}