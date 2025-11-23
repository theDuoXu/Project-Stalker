package projectstalker.benchmark;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import projectstalker.config.RiverConfig;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;
import projectstalker.factory.RiverGeometryFactory;
import projectstalker.physics.i.ITransportSolver;
import projectstalker.physics.impl.SplitOperatorTransportSolver;
import projectstalker.physics.jni.GpuMusclTransportSolver;

import java.util.Arrays;

@Tag("Benchmark")
@Slf4j
public class TransportGpuBenchmark {

    private RiverGeometry geometry;
    private RiverState initialState;

    private ITransportSolver cpuSolver;
    private ITransportSolver gpuSolver;

    // Constantes para forzar carga de trabajo
    private final int CELL_COUNT = 10_000; // Un río largo (500km a 50m/celda)
    private final float VELOCITY = 2.0f;   // Río rápido para forzar dt pequeño (muchos pasos)

    @BeforeEach
    void setUp() {
        // 1. Geometría Grande (10k celdas para saturar la GPU)
        RiverConfig config = RiverConfig.builder()
                .totalLength(CELL_COUNT * 50.0f)
                .spatialResolution(50.0f)
                .baseWidth(50.0f)
                .averageSlope(0.001f)
                .baseManning(0.03f)
                .baseDecayRateAt20C(0.0001f)
                .baseDispersionAlpha(5.0f)
                .build();

        RiverGeometryFactory factory = new RiverGeometryFactory();
        this.geometry = factory.createRealisticRiver(config);

        // 2. Solvers
        // Forzamos CFL 0.9 en ambos para que hagan el mismo número de pasos
        this.cpuSolver = new SplitOperatorTransportSolver(
                new projectstalker.physics.impl.MusclAdvectionSolver(),
                new projectstalker.physics.impl.CentralDiffusionSolver(),
                new projectstalker.physics.impl.FirstOrderReactionSolver(),
                0.9
        );

        try {
            this.gpuSolver = new GpuMusclTransportSolver();
        } catch (UnsatisfiedLinkError e) {
            log.error("No se pudo cargar GPU Solver. El benchmark fallará.", e);
        }

        // 3. Estado Inicial (Mancha en el centro)
        float[] h = new float[geometry.getCellCount()]; Arrays.fill(h, 2.0f);
        float[] u = new float[geometry.getCellCount()]; Arrays.fill(u, VELOCITY);
        float[] c = new float[geometry.getCellCount()];
        int mid = geometry.getCellCount() / 2;
        for(int i=0; i<100; i++) c[mid+i] = 100.0f; // Mancha grande

        this.initialState = RiverState.builder()
                .waterDepth(h)
                .velocity(u)
                .contaminantConcentration(c)
                .temperature(new float[geometry.getCellCount()])
                .ph(new float[geometry.getCellCount()])
                .build();

        log.info("Setup Benchmark: {} celdas. Velocidad={} m/s.", geometry.getCellCount(), VELOCITY);
    }

    @Test
    @DisplayName("Benchmark: Evolución Temporal (CPU vs GPU)")
    void benchmarkTimeEvolution() {
        log.info("=== INICIANDO BENCHMARK DE TRANSPORTE ===");

        // Duraciones a simular (en segundos)
        // 1h, 6h, 24h, 1 semana
        float[] durations = {3600f, 21600f, 86400f, 604800f};

        // --- WARM-UP ---
        log.info(">> Calentando motores...");
        runIteration(cpuSolver, 100f);
        runIteration(gpuSolver, 100f);
        log.info(">> Calentamiento completado.\n");

        System.out.printf("%-15s | %-15s | %-15s | %-15s%n", "SIM DURATION", "CPU (ms)", "GPU (ms)", "SPEEDUP");
        System.out.println("---------------------------------------------------------------------");

        for (float duration : durations) {
            System.gc(); // Limpieza para no medir GC

            // 1. Medir CPU
            // Nota: Para duraciones muy largas (1 semana), la CPU podría tardar minutos.
            // Si ves que tarda demasiado, comenta la línea de CPU para duraciones extremas.
            double cpuTimeMs = runIteration(cpuSolver, duration);

            // 2. Medir GPU
            double gpuTimeMs = runIteration(gpuSolver, duration);

            // 3. Reportar
            double speedup = cpuTimeMs / gpuTimeMs;
            String label = formatDuration(duration);

            System.out.printf("%-15s | %-15.2f | %-15.2f | %-15.2fx %s%n",
                    label, cpuTimeMs, gpuTimeMs, speedup,
                    (speedup > 5.0 ? "🚀" : ""));
        }
    }

    private double runIteration(ITransportSolver solver, float duration) {
        long start = System.nanoTime();

        // La magia ocurre aquí:
        // El solver dividirá 'duration' en miles de micro-pasos (sub-stepping).
        // La CPU hará un bucle Java. La GPU hará un bucle C++ lanzando kernels.
        solver.solve(initialState, geometry, duration);

        long end = System.nanoTime();
        return (end - start) / 1_000_000.0;
    }

    private String formatDuration(float seconds) {
        if (seconds >= 86400) return String.format("%.1f Days", seconds / 86400);
        if (seconds >= 3600) return String.format("%.1f Hours", seconds / 3600);
        return String.format("%.0f Secs", seconds);
    }
}