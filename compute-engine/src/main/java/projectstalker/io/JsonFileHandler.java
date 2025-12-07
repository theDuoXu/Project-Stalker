package projectstalker.io;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import lombok.extern.slf4j.Slf4j;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Gestiona la serialización (escritura) y deserialización (lectura) de objetos
 * hacia y desde archivos JSON.
 * <p>
 * Esta clase es genérica y puede trabajar con cualquier tipo de objeto (POJO)
 * que sea compatible con la librería Jackson.
 */
@Slf4j
public class JsonFileHandler {

    // 1. El ObjectMapper es el motor de Jackson.
    // Es costoso de crear, así que lo creamos y lo reutilizamos.
    // Es thread-safe.
    private static final ObjectMapper objectMapper = createConfiguredObjectMapper();

    private static ObjectMapper createConfiguredObjectMapper() {
        ObjectMapper mapper = new ObjectMapper();
        // 2. Habilita la indentación para que los archivos JSON sean legibles por humanos.
        mapper.enable(SerializationFeature.INDENT_OUTPUT);
        // 3. Registra módulos para tipos modernos de Java, como fechas de Java 8.
        mapper.findAndRegisterModules();
        return mapper;
    }

    /**
     * Serializa un objeto a un archivo JSON en la ruta especificada.
     * Si el archivo ya existe, será sobrescrito.
     *
     * @param data El objeto a serializar. No puede ser nulo.
     * @param filePath La ruta completa del archivo de destino (ej: "data/geometries/rio_ebro.json").
     * @param <T> El tipo del objeto a serializar.
     * @throws IOException Si ocurre un error durante la escritura del archivo.
     */
    public <T> void writeToFile(T data, String filePath) throws IOException {
        Path path = Paths.get(filePath);
        log.info("Serializando objeto de tipo {} a archivo: {}", data.getClass().getSimpleName(), path.toAbsolutePath());

        try {
            // Asegurarse de que el directorio padre existe
            Files.createDirectories(path.getParent());
            // Jackson hace la magia de convertir el objeto a un String JSON y escribirlo.
            objectMapper.writeValue(path.toFile(), data);
            log.debug("Escritura a JSON completada con éxito.");
        } catch (IOException e) {
            log.error("Error fatal al escribir el archivo JSON en {}", path.toAbsolutePath(), e);
            throw e; // Relanzamos la excepción para que el llamador pueda manejarla.
        }
    }

    /**
     * Deserializa un archivo JSON para reconstruir un objeto de un tipo específico.
     *
     * @param filePath La ruta del archivo JSON a leer.
     * @param objectType El tipo de clase al que se debe convertir el JSON (ej: RiverGeometry.class).
     * @param <T> El tipo del objeto a deserializar.
     * @return Una nueva instancia del objeto reconstruido desde el JSON.
     * @throws IOException Si el archivo no se encuentra o hay un error de lectura o formato.
     */
    public <T> T readFromFile(String filePath, Class<T> objectType) throws IOException {
        Path path = Paths.get(filePath);
        log.info("Deserializando archivo {} a un objeto de tipo {}", path.toAbsolutePath(), objectType.getSimpleName());

        if (!Files.exists(path)) {
            throw new IOException("El archivo especificado no existe: " + path.toAbsolutePath());
        }

        try {
            // Jackson lee el archivo y usa la Class<T> para saber cómo construir el objeto.
            return objectMapper.readValue(path.toFile(), objectType);
        } catch (IOException e) {
            log.error("Error fatal al leer o parsear el archivo JSON desde {}", path.toAbsolutePath(), e);
            throw e;
        }
    }
}