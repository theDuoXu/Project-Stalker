package projectstalker.physics.jni;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import projectstalker.config.RiverConfig;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;
import projectstalker.factory.RiverGeometryFactory;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

@Tag("GPU")
@Slf4j
class GpuMusclTransportIntegrationTest {

    private GpuMusclTransportSolver solver; // SUT Real
    private RiverGeometry realGeometry;
    private int cellCount;

    @BeforeEach
    void setUp() {
        log.info("Configurando Test de Integración de Transporte GPU...");

        // 1. Crear Geometría Real (Pequeña para que sea rápido)
        // Usamos un río de prueba simple
        RiverConfig config = RiverConfig.builder()
                .totalLength(1000) // 1 km
                .spatialResolution(10.0f) // 10m
                .baseWidth(10.0f)
                .averageSlope(0.001f)
                .build();

        RiverGeometryFactory factory = new RiverGeometryFactory();
        this.realGeometry = factory.createRealisticRiver(config);
        this.cellCount = this.realGeometry.getCellCount();

        // 2. Instanciar el Solver Real (Cargará la librería)
        try {
            // Usamos el constructor por defecto que usa el Singleton
            this.solver = new GpuMusclTransportSolver();
            log.info("GpuMusclTransportSolver instanciado. Librería nativa cargada.");
        } catch (UnsatisfiedLinkError e) {
            fail("No se pudo cargar la librería nativa. Asegúrate de ejecutar con la tarea 'gpuTest'. Error: " + e.getMessage());
        }
    }

    @Test
    @DisplayName("Smoke Test: El solver GPU debe ejecutar un paso sin errores y devolver un estado válido")
    void solve_shouldRunOnGpuWithoutCrashing() {
        log.info(">>> TEST: Ejecución Nativa de Transporte (Smoke Test)");

        // ARRANGE
        // Estado Inicial: Agua fluyendo, Mancha en el centro
        float[] h = new float[cellCount]; Arrays.fill(h, 1.0f);
        float[] u = new float[cellCount]; Arrays.fill(u, 1.0f);
        float[] temp = new float[cellCount]; Arrays.fill(temp, 20.0f);
        float[] ph = new float[cellCount]; Arrays.fill(ph, 7.0f);

        float[] c = new float[cellCount];
        // Ponemos un pico de concentración en el medio
        int mid = cellCount / 2;
        c[mid] = 100.0f;

        RiverState state0 = new RiverState(h, u, c, temp, ph);

        float dt = 1.0f; // Paso de tiempo pequeño

        // ACT
        // Llamada real a la GPU (JNI -> C++ -> CUDA -> Kernel)
        RiverState state1 = assertDoesNotThrow(() ->
                        solver.solve(state0, realGeometry, dt),
                "La llamada al solver nativo falló (Crash/Excepción)."
        );

        // ASSERT
        assertNotNull(state1, "El estado resultante no debe ser nulo.");

        // Verificaciones básicas de cordura
        float[] cNew = state1.contaminantConcentration();
        assertEquals(cellCount, cNew.length, "El array de concentración devuelto tiene tamaño incorrecto.");

        // Verificar que la mancha sigue ahí (aproximadamente)
        // Con u=1, dt=1, dx=10, el pico debería moverse muy poco (0.1 celda), así que el máximo debe seguir cerca del centro.
        // Y por difusión, el pico debe haber bajado un poco de 100.
        float maxVal = 0.0f;
        for(float val : cNew) maxVal = Math.max(maxVal, val);

        log.info("Pico inicial: 100.0 -> Pico final: {}", maxVal);

        assertTrue(maxVal > 0.0f, "El contaminante desapareció (Todo ceros).");
        assertTrue(maxVal <= 100.0f, "Violación TVD: El contaminante creció mágicamente.");
        assertFalse(Float.isNaN(maxVal), "El resultado contiene NaNs.");

        log.info("¡ÉXITO! La GPU procesó el transporte correctamente.");
    }
}