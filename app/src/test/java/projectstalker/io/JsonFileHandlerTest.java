package projectstalker.io;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir; // Inyección de un directorio temporal para las pruebas

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat; // Usamos AssertJ para aserciones más legibles
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Pruebas unitarias para la clase {@link JsonFileHandler}.
 * Estas pruebas verifican la correcta serialización y deserialización de objetos a/desde archivos JSON.
 */
class JsonFileHandlerTest {

    // La instancia de la clase que vamos a probar
    private JsonFileHandler jsonFileHandler;

    // JUnit 5 inyectará un directorio temporal fresco antes de cada test.
    // Esto es CRUCIAL para que las pruebas no interfieran entre sí y no dejen basura.
    @TempDir
    Path tempDir;

    // Clase de datos de prueba. Un 'record' es perfecto porque ya tiene
    // `equals()`, `hashCode()` y `toString()` implementados correctamente.
    private record TestData(String name, int value, List<String> items) {}

    @BeforeEach
    void setUp() {
        // Creamos una nueva instancia antes de cada test para asegurar el aislamiento.
        this.jsonFileHandler = new JsonFileHandler();
    }

    @Test
    @DisplayName("Debería serializar un objeto a un archivo JSON correctamente")
    void writeToFile_shouldCreateAndWriteToJsonFile() throws IOException {
        // --- 1. Arrange (Preparación) ---
        TestData originalData = new TestData("TestObject", 123, List.of("A", "B", "C"));
        Path outputFile = tempDir.resolve("output.json");

        // --- 2. Act (Actuación) ---
        jsonFileHandler.writeToFile(originalData, outputFile.toString());

        // --- 3. Assert (Verificación) ---
        // La aserción más importante: el archivo existe.
        assertThat(outputFile).exists();

        // Verificamos que el contenido del archivo es el JSON que esperamos.
        String fileContent = Files.readString(outputFile);
        assertThat(fileContent)
                .contains("\"name\" : \"TestObject\"")
                .contains("\"value\" : 123")
                .contains("\"items\" : [ \"A\", \"B\", \"C\" ]");
    }

    @Test
    @DisplayName("Debería deserializar un archivo JSON a un objeto correctamente")
    void readFromFile_shouldReadAndParseJsonFile() throws IOException {
        // --- 1. Arrange ---
        // Creamos manualmente el archivo JSON que la prueba leerá.
        String jsonContent = """
        {
          "name": "ExpectedObject",
          "value": 456,
          "items": ["X", "Y"]
        }
        """;
        Path inputFile = tempDir.resolve("input.json");
        Files.writeString(inputFile, jsonContent);

        TestData expectedData = new TestData("ExpectedObject", 456, List.of("X", "Y"));

        // --- 2. Act ---
        TestData resultData = jsonFileHandler.readFromFile(inputFile.toString(), TestData.class);

        // --- 3. Assert ---
        // Usamos la comparación de objetos de AssertJ. Es mucho más potente que assertEquals.
        assertThat(resultData).isNotNull();
        assertThat(resultData).isEqualTo(expectedData);
    }

    @Test
    @DisplayName("Debería completar un ciclo de escritura y lectura (round-trip) con éxito")
    void writeAndRead_shouldResultInEqualObjects() throws IOException {
        // --- 1. Arrange ---
        TestData originalData = new TestData("RoundTrip", 999, List.of("Start", "End"));
        Path testFile = tempDir.resolve("roundtrip.json");

        // --- 2. Act ---
        jsonFileHandler.writeToFile(originalData, testFile.toString());
        TestData readData = jsonFileHandler.readFromFile(testFile.toString(), TestData.class);

        // --- 3. Assert ---
        assertThat(readData).isEqualTo(originalData);
    }

    @Test
    @DisplayName("Debería lanzar IOException al intentar leer un archivo que no existe")
    void readFromFile_whenFileDoesNotExist_shouldThrowIOException() {
        // --- 1. Arrange ---
        Path nonExistentFile = tempDir.resolve("imaginary.json");

        // --- 2. Act & 3. Assert ---
        // Verificamos que se lanza la excepción esperada cuando se ejecuta el código.
        IOException exception = assertThrows(IOException.class, () -> {
            jsonFileHandler.readFromFile(nonExistentFile.toString(), TestData.class);
        });

        // Opcional: podemos verificar el mensaje de la excepción para ser más precisos.
        assertThat(exception.getMessage()).contains("El archivo especificado no existe");
    }

    @Test
    @DisplayName("Debería lanzar IOException al intentar leer un archivo JSON mal formado")
    void readFromFile_whenJsonIsMalformed_shouldThrowIOException() throws IOException {
        // --- 1. Arrange ---
        // Creamos un JSON con un error de sintaxis (una coma extra al final).
        String malformedJson = """
        {
          "name": "Malformed",
          "value": 100,
        }
        """;
        Path inputFile = tempDir.resolve("malformed.json");
        Files.writeString(inputFile, malformedJson);

        // --- 2. Act & 3. Assert ---
        assertThrows(IOException.class, () -> {
            jsonFileHandler.readFromFile(inputFile.toString(), TestData.class);
        });
    }
}