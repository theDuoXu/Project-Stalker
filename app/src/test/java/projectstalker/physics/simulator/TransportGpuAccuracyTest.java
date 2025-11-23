package projectstalker.physics.simulator;

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

import static org.junit.jupiter.api.Assertions.*;

@Tag("GPU")
@Slf4j
class TransportGpuAccuracyTest {

    private ITransportSolver cpuSolver;
    private ITransportSolver gpuSolver;
    private RiverGeometry geometry;
    private int cellCount;

    private final float EPSILON = 1e-3f;

    @BeforeEach
    void setUp() {
        // 1. Configuración
        RiverConfig config = RiverConfig.builder()
                .totalLength(500)
                .spatialResolution(10.0f) // dx = 10.0
                .baseWidth(20.0f)
                .averageSlope(0.001f)
                .baseManning(0.03f)
                .baseDecayRateAt20C(0.0f)
                .baseDispersionAlpha(0.0f) // Sin dispersión para testear transporte puro
                .build();

        RiverGeometryFactory factory = new RiverGeometryFactory();
        this.geometry = factory.createRealisticRiver(config);
        this.cellCount = geometry.getCellCount();

        // Verificación de Seguridad
        log.info("Geometría: Celdas={}, DX={}", cellCount, this.geometry.getSpatialResolution());
        if (this.geometry.getSpatialResolution() < 0.1f) {
            throw new IllegalStateException("DX es 0. Revisa RiverGeometryFactory.");
        }

        // 2. Solvers
        // Aseguramos que el CFL no sea 0 pasando el valor explícito al constructor
        this.cpuSolver = new SplitOperatorTransportSolver(
                new projectstalker.physics.impl.MusclAdvectionSolver(),
                new projectstalker.physics.impl.CentralDiffusionSolver(),
                new projectstalker.physics.impl.FirstOrderReactionSolver(),
                0.9 // <--- FORZAMOS CFL 0.9 PARA EVITAR BUCLE INFINITO
        );

        try {
            this.gpuSolver = new GpuMusclTransportSolver();
        } catch (UnsatisfiedLinkError e) {
            fail("Error librería nativa: " + e.getMessage());
        }
    }

    @Test
    @DisplayName("Paridad: Advección Pura de Onda Cuadrada")
    void compare_advection_squareWave() {
        log.info(">>> INICIO TEST PARIDAD <<<");

        // 1. Datos Iniciales
        float[] h = new float[cellCount]; Arrays.fill(h, 2.0f);
        float[] u = new float[cellCount]; Arrays.fill(u, 1.0f); // 1 m/s
        float[] c = new float[cellCount];

        // Onda cuadrada en el centro (indices 20-30)
        int start = cellCount / 2 - 5;
        for(int i=0; i<10; i++) c[start + i] = 100.0f;

        // USO DEL BUILDER: Previene errores de orden de argumentos
        RiverState initialState = RiverState.builder()
                .waterDepth(h)
                .velocity(u)
                .contaminantConcentration(c)
                .temperature(new float[cellCount]) // Ceros
                .ph(new float[cellCount])          // Ceros
                .build();

        float dt = 5.0f; // Debería resultar en 1 solo paso si CFL=0.9

        // 2. Ejecución
        RiverState resCpu = cpuSolver.solve(initialState, geometry, dt);
        RiverState resGpu = gpuSolver.solve(initialState, geometry, dt);

        // 3. Validación
        compareStates(resCpu, resGpu);
    }

    private void compareStates(RiverState cpu, RiverState gpu) {
        float[] cCpu = cpu.contaminantConcentration();
        float[] cGpu = gpu.contaminantConcentration();

        double maxDiff = 0.0;

        // Saltamos bordes para evitar ruido de condiciones de frontera
        for (int i = 1; i < cellCount - 1; i++) {
            double diff = Math.abs(cCpu[i] - cGpu[i]);
            if (diff > maxDiff) maxDiff = diff;

            if (diff > EPSILON) {
                log.error("Divergencia Celda {}: CPU={} vs GPU={}", i, cCpu[i], cGpu[i]);
                fail(String.format("Divergencia celda %d. Diff=%.4f", i, diff));
            }
        }
        log.info("Máxima diferencia (Interior): {}", maxDiff);
        log.info("¡PARIDAD CONFIRMADA!");
    }
}