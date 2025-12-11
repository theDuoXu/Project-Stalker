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
import projectstalker.physics.solver.TransportSolver;
import projectstalker.physics.solver.impl.CentralDiffusionSolver;
import projectstalker.physics.solver.impl.FirstOrderReactionSolver;
import projectstalker.physics.solver.impl.MusclAdvectionSolver;
import projectstalker.physics.solver.impl.SplitOperatorTransportSolver;
import projectstalker.physics.jni.GpuMusclTransportSolver;

import java.util.Arrays;

@Tag("Benchmark")
@Slf4j
public class TransportGpuBenchmark {

    private RiverGeometry geometry;
    private RiverState initialState;

    private TransportSolver cpuSolver;
    private TransportSolver gpuSolver;

    // --- CONFIGURACIÓN DE CARGA ---
    // 1 Millón de celdas para saturar la GPU y ver el speedup real
    private final int CELL_COUNT = 1_000_000;
    private final float VELOCITY = 2.0f;

    // Umbral para dejar de ejecutar CPU real e interpolar
    private final int CPU_EXECUTION_THRESHOLD = 50_000;

    @BeforeEach
    void setUp() {
        // 1. Geometría Grande
        RiverConfig config = RiverConfig.builder()
                .totalLength(CELL_COUNT * 50.0f) // 50m * 1M = 50,000 km (Río sintético masivo)
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
        // Forzamos CFL 0.9 en ambos para garantizar paridad algorítmica
        this.cpuSolver = new SplitOperatorTransportSolver(
                new MusclAdvectionSolver(),
                new CentralDiffusionSolver(),
                new FirstOrderReactionSolver(),
                0.9
        );

        try {
            this.gpuSolver = new GpuMusclTransportSolver();
        } catch (UnsatisfiedLinkError e) {
            log.error("No se pudo cargar GPU Solver. El benchmark fallará.", e);
        }

        // 3. Estado Inicial
        float[] h = new float[geometry.getCellCount()]; Arrays.fill(h, 2.0f);
        float[] u = new float[geometry.getCellCount()]; Arrays.fill(u, VELOCITY);
        float[] c = new float[geometry.getCellCount()];

        // Mancha en el centro
        int mid = geometry.getCellCount() / 2;
        int blobSize = Math.min(1000, geometry.getCellCount() / 10);
        for(int i=0; i<blobSize; i++) c[mid+i] = 100.0f;

        this.initialState = RiverState.builder()
                .waterDepth(h)
                .velocity(u)
                .contaminantConcentration(c)
                .temperature(new float[geometry.getCellCount()])
                .ph(new float[geometry.getCellCount()])
                .build();

        log.info("Setup Benchmark: {} celdas. Velocidad={} m/s. (Límite CPU real: {} celdas)",
                String.format("%,d", geometry.getCellCount()), VELOCITY, CPU_EXECUTION_THRESHOLD);
    }

    @Test
    @DisplayName("Benchmark: Evolución Temporal (CPU Interpolada vs GPU Real)")
    void benchmarkTimeEvolution() {
        log.info("=== INICIANDO BENCHMARK MASIVO (1M Celdas) ===");

        // Duraciones: 1h, 6h, 24h, 1 semana
        float[] durations = {3600f, 21600f, 86400f, 604800f};

        // --- WARM-UP (Corto) ---
        // Necesario para JIT y Contexto CUDA
        log.info(">> Calentando motores (100 iteraciones)...");
        runIteration(cpuSolver, 10.0f); // Muy corto para no esperar
        runIteration(gpuSolver, 10.0f);
        log.info(">> Calentamiento completado.\n");

        System.out.printf("%-15s | %-20s | %-15s | %-15s%n", "SIM DURATION", "CPU (ms)", "GPU (ms)", "SPEEDUP");
        System.out.println("----------------------------------------------------------------------------");

        // Variables para interpolación CPU
        double cpuBaselineMs = 0;
        float baselineDuration = durations[0];

        for (int i = 0; i < durations.length; i++) {
            float duration = durations[i];
            System.gc();

            // 1. Lógica CPU: Ejecutar o Estimar
            double cpuTimeMs;
            boolean isCpuEstimated = false;

            // Si tenemos muchas celdas, solo ejecutamos la primera duración (la más corta)
            // y usamos esa velocidad para proyectar el resto.
            if (CELL_COUNT > CPU_EXECUTION_THRESHOLD && i > 0) {
                // Interpolación lineal: T = T_base * (Duration / Duration_base)
                cpuTimeMs = cpuBaselineMs * (duration / baselineDuration);
                isCpuEstimated = true;
            } else {
                // Ejecución Real
                if (CELL_COUNT > CPU_EXECUTION_THRESHOLD) {
                    log.info("Ejecutando CPU real para base line ({}s)... esto puede tardar un poco.", duration);
                }
                cpuTimeMs = runIteration(cpuSolver, duration);

                // Guardamos la base si es la primera iteración
                if (i == 0) {
                    cpuBaselineMs = cpuTimeMs;
                    baselineDuration = duration;
                }
            }

            // 2. Medir GPU (Siempre Real)
            // La GPU debe aguantar el millón de celdas sin interpolar
            double gpuTimeMs = runIteration(gpuSolver, duration);

            // 3. Reportar
            double speedup = cpuTimeMs / gpuTimeMs;
            String label = formatDuration(duration);
            String cpuLabel = String.format("%,.2f %s", cpuTimeMs, isCpuEstimated ? "(Est.)" : "");

            System.out.printf("%-15s | %-20s | %-15.2f | %-15.2fx %s%n",
                    label, cpuLabel, gpuTimeMs, speedup,
                    (speedup > 10.0 ? "MUY BIEN" : (speedup > 5.0 ? "BIEN" : "")));
        }
    }

    private double runIteration(TransportSolver solver, float duration) {
        long start = System.nanoTime();
        solver.solve(initialState, geometry, duration);
        long end = System.nanoTime();
        return (end - start) / 1_000_000.0;
    }

    private String formatDuration(float seconds) {
        if (seconds >= 604800) return String.format("%.0f Week", seconds / 604800);
        if (seconds >= 86400) return String.format("%.1f Days", seconds / 86400);
        if (seconds >= 3600) return String.format("%.1f Hours", seconds / 3600);
        return String.format("%.0f Secs", seconds);
    }
}