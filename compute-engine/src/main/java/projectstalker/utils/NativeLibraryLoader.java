package projectstalker.utils;

import lombok.extern.slf4j.Slf4j;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;

@Slf4j
public class NativeLibraryLoader {

    /**
     * Carga una librería nativa embebida en el JAR.
     * @param libraryName El nombre base sin prefijo 'lib' ni extensión '.so'.
     */
    public static void loadLibrary(String libraryName) {
        try {
            // 1. Detectar nombre del archivo según SO
            String os = System.getProperty("os.name").toLowerCase();
            String pathInJar;

            if (os.contains("linux")) {
                pathInJar = "/native/linux-x86_64/lib" + libraryName + ".so";
            } else if (os.contains("win")) {
                pathInJar = "/native/windows-x86_64/" + libraryName + ".dll";
            } else {
                throw new UnsupportedOperationException("SO no soportado: " + os);
            }

            log.info(">>> JNI: Intentando extraer y cargar: {}", pathInJar);

            // 2. Obtener stream desde el JAR
            InputStream in = NativeLibraryLoader.class.getResourceAsStream(pathInJar);
            if (in == null) {
                throw new IOException("No se encontró la librería dentro del JAR en: " + pathInJar);
            }

            // 3. Crear archivo temporal en el disco real
            File tempFile = File.createTempFile("lib" + libraryName, os.contains("win") ? ".dll" : ".so");
            tempFile.deleteOnExit(); // Limpieza automática al cerrar la app

            // 4. Copiar del JAR al TMP
            Files.copy(in, tempFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
            in.close();

            // 5. Cargar desde el archivo temporal (Ruta absoluta)
            System.load(tempFile.getAbsolutePath());

            log.info(">>> JNI: Librería cargada exitosamente desde: {}", tempFile.getAbsolutePath());

        } catch (IOException e) {
            log.error(">>> JNI: Error de E/S extrayendo librería.", e);
            throw new RuntimeException(e);
        } catch (UnsatisfiedLinkError e) {
            log.error(">>> JNI: Error al vincular la librería nativa.", e);
            throw e;
        }
    }
}