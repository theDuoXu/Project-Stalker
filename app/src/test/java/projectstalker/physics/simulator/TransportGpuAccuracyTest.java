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

    // Tolerancia: 1e-3 es razonable para diferencias acumuladas float vs double
    private final float EPSILON = 1e-3f;

    @BeforeEach
    void setUp() {
        // 1. Geometría Controlada
        RiverConfig config = RiverConfig.builder()
                .totalLength(500)
                .spatialResolution(10.0f)
                .baseWidth(20.0f)
                .averageSlope(0.001f)
                .baseManning(0.03f)
                .baseDecayRateAt20C(0.0f) // Sin reacción inicial
                .baseDispersionAlpha(0.0f) // Sin dispersión inicial
                .build();

        RiverGeometryFactory factory = new RiverGeometryFactory();
        this.geometry = factory.createRealisticRiver(config);
        this.cellCount = geometry.getCellCount();

        // VALIDACIÓN DE SEGURIDAD
        if (this.geometry.getSpatialResolution() <= 0.001) {
            throw new IllegalStateException("¡ERROR CRÍTICO! La geometría se creó con resolución 0.");
        }

        this.cpuSolver = new SplitOperatorTransportSolver();

        try {
            this.gpuSolver = new GpuMusclTransportSolver();
        } catch (UnsatisfiedLinkError e) {
            fail("No se pudo cargar la librería GPU. Ejecuta con ./gradlew gpuTest");
        }

        log.info("Setup completo. Celdas: {}", cellCount);
    }

    @Test
    @DisplayName("Paridad: Advección Pura de Onda Cuadrada (CPU vs GPU)")
    void compare_advection_squareWave() {
        log.info(">>> INICIO TEST PARIDAD <<<");

        // 1. DATOS LIMPIOS
        float[] h = new float[cellCount]; Arrays.fill(h, 2.0f);
        float[] u = new float[cellCount]; Arrays.fill(u, 1.0f);
        float[] t = new float[cellCount]; Arrays.fill(t, 0.0f); // Temp 0 (Neutro)
        float[] ph = new float[cellCount]; Arrays.fill(ph, 0.0f); // pH 0 (Neutro)

        float[] c = new float[cellCount];
        // Onda cuadrada de 10 celdas en el centro
        int start = cellCount / 2 - 5;
        for(int i=0; i<10; i++) c[start + i] = 100.0f;

        // USO DEL BUILDER: Evita errores de orden en los argumentos
        RiverState initialState = RiverState.builder()
                .waterDepth(h)
                .velocity(u)
                .contaminantConcentration(c) // Explícito e inequívoco
                .temperature(t)
                .ph(ph)
                .build();

        // Simulamos 5 segundos (Media celda de movimiento)
        float dt = 5.0f;

        // 2. EJECUCIÓN
        RiverState resCpu = cpuSolver.solve(initialState, geometry, dt);
        RiverState resGpu = gpuSolver.solve(initialState, geometry, dt);

        // 3. COMPARACIÓN
        compareStates(resCpu, resGpu);
    }

    private void compareStates(RiverState cpu, RiverState gpu) {
        float[] cCpu = cpu.contaminantConcentration();
        float[] cGpu = gpu.contaminantConcentration();

        double maxDiff = 0.0;

        // IMPORTANTE: Saltamos las celdas frontera (0 y N-1)
        // Las condiciones de frontera (Dirichlet vs Neumann) pueden tener ligeras
        // diferencias de implementación numérica. Validamos el transporte interior.
        for (int i = 1; i < cellCount - 1; i++) {

            double diff = Math.abs(cCpu[i] - cGpu[i]);
            if (diff > maxDiff) maxDiff = diff;

            if (diff > EPSILON) {
                log.error("Divergencia Celda {}: CPU={} vs GPU={}", i, cCpu[i], cGpu[i]);
                fail(String.format("Divergencia en celda %d. Diff=%.4f", i, diff));
            }
        }
        log.info("Máxima diferencia encontrada (Interior): {}", maxDiff);
        log.info("¡PARIDAD CONFIRMADA!");
    }
}