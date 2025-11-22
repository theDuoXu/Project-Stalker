package projectstalker.benchmark;

import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.physics.impl.FirstOrderReactionSolver;

import java.util.concurrent.TimeUnit;
import java.util.Random;

import static org.mockito.Mockito.*;

/**
 * Clase de Microbenchmark JMH para comparar el rendimiento
 * entre la ejecución secuencial (for-loop) y la ejecución paralela
 * (Parallel Stream) del solver de reacciones de primer orden.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Thread)
public class ReactionSolverBenchmark {

    // --- PARAMETRIZACIÓN DEL BENCHMARK ---
    // JMH ejecutará los tests para cada uno de estos tamaños de array (N).
    // Usamos el umbral (10k) como punto clave.
    @Param({"1000", "10000", "100000", "1000000"})
    private int size;

    private FirstOrderReactionSolver solverSequential;
    private FirstOrderReactionSolver solverParallel;

    // Arrays que serán rellenados con datos aleatorios
    private float[] concentration;
    private float[] temperature;

    // Dependencias
    private RiverGeometry mockGeometry;
    private float dt = 0.5f;

    /**
     * Configuración que se ejecuta una vez por cada hilo de prueba.
     * Aquí se inicializan los datos y los Solvers.
     */
    @Setup(Level.Trial)
    public void setup() {
        // 1. Inicialización de los arrays de datos
        concentration = new float[size];
        temperature = new float[size];
        Random rand = new Random(42); // Semilla fija para reproducibilidad

        for (int i = 0; i < size; i++) {
            concentration[i] = 10.0f + rand.nextFloat() * 50.0f; // C entre 10 y 60
            temperature[i] = 15.0f + rand.nextFloat() * 10.0f; // T entre 15 y 25 grados
        }

        // 2. Mock de RiverGeometry
        mockGeometry = mock(RiverGeometry.class);

        // Mockeamos el coeficiente de decaimiento base (k20) para que sea constante en todas las celdas
        // Esto simplifica el benchmarking al asegurar que todas las celdas tienen la misma carga computacional.
        when(mockGeometry.getBaseDecayAt(anyInt())).thenReturn(0.25f);

        // 3. Instanciación de los Solvers
        // El solver secuencial se fuerza a NO usar paralelismo, ignorando su umbral interno.
        solverSequential = new FirstOrderReactionSolver(1.047, 20.0, false);

        // El solver paralelo activa el uso de paralelismo y dejará que el 'if (n > 10_000)' decida
        solverParallel = new FirstOrderReactionSolver(1.047, 20.0, true);
    }

    /**
     * Test para medir el rendimiento de la ejecución secuencial (bucle for).
     */
    @Benchmark
    public void testSequential(Blackhole bh) {
        // Ejecutamos el método del solver y pasamos el resultado al Blackhole
        bh.consume(solverSequential.solveReaction(concentration, temperature, mockGeometry, dt));
    }

    /**
     * Test para medir el rendimiento de la ejecución paralela (IntStream.parallel()).
     */
    @Benchmark
    public void testParallel(Blackhole bh) {
        // Ejecutamos el método del solver. Si 'size' es > 10000, usará ForkJoinPool.
        bh.consume(solverParallel.solveReaction(concentration, temperature, mockGeometry, dt));
    }
}