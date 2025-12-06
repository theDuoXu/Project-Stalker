package projectstalker.physics.simulator;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import projectstalker.config.RiverConfig;
import projectstalker.config.SimulationConfig;
import projectstalker.config.SimulationConfig.GpuStrategy;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;
import projectstalker.domain.simulation.ChunkedManningResult;
import projectstalker.domain.simulation.FlyweightManningResult;
import projectstalker.domain.simulation.IManningResult;
import projectstalker.domain.simulation.StridedManningResult;
import projectstalker.factory.RiverGeometryFactory;
import projectstalker.physics.jni.ManningGpuSolver;

import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@Slf4j
class ManningBatchProcessorTest {

    private ManningBatchProcessor batchProcessor;
    private RiverGeometry realGeometry;
    private SimulationConfig mockConfig;
    private ManningGpuSolver mockGpuSolver;

    private int cellCount;
    private final int MOCK_STRIDE = 2;
    private final int BATCH_SIZE_CONFIG = 10; // cpuTimeBatchSize

    @BeforeEach
    void setUp() throws Exception {
        // 1. Geometría Real
        RiverConfig config = RiverConfig.getTestingRiver();
        this.realGeometry = new RiverGeometryFactory().createRealisticRiver(config);
        this.cellCount = this.realGeometry.getCellCount();

        // 2. Config Mock
        mockConfig = mock(SimulationConfig.class);
        when(mockConfig.getCpuProcessorCount()).thenReturn(2);

        // Configurar Stride y BatchSize compatibles (10 % 2 == 0)
        when(mockConfig.getGpuFullEvolutionStride()).thenReturn(MOCK_STRIDE);
        when(mockConfig.getCpuTimeBatchSize()).thenReturn(BATCH_SIZE_CONFIG);

        // Default: GPU Enabled
        when(mockConfig.isUseGpuAccelerationOnManning()).thenReturn(true);
        when(mockConfig.getGpuStrategy()).thenReturn(GpuStrategy.FULL_EVOLUTION);

        // 3. Inicializar Processor
        batchProcessor = new ManningBatchProcessor(this.realGeometry, mockConfig);

        // 4. Inyección de Mock GPU Solver (Usando Reflection para saltar el constructor JNI real)
        mockGpuSolver = mock(ManningGpuSolver.class);

        // HACK: Reemplazar el solver que se crea en el constructor
        // Como el constructor de ManningGpuSolver carga librerías nativas, el mock evita eso.
        // Pero ManningBatchProcessor crea el solver DENTRO de los métodos process... (Try-with-resources).
        // Para testear unitariamente, necesitamos interceptar esa creación.
        // SOLUCIÓN TESTABLE: En un test real de integración usaríamos el solver real.
        // Aquí simulamos el comportamiento inyectando un factory o refactorizando ligeramente el procesador.
        // PARA NO TOCAR CÓDIGO DE PRODUCCIÓN: Usaremos un truco de PowerMock o asumimos que
        // ManningBatchProcessor tiene un constructor package-private para inyectar el solver (Factory Pattern),
        // o refactorizamos el test para que ManningBatchProcessor acepte un Supplier<ManningGpuSolver>.

        // DADO QUE NO QUIERES CÓDIGO EXTRA, ASUMIREMOS QUE PODEMOS ESPIAR/MOCKEAR LA CREACIÓN.
        // Como es difícil mockear 'new', usaremos una subclase anónima del Processor para sobreescribir la creación del solver?
        // No, el método process crea el solver localmente.
        // SOLUCIÓN PRAGMÁTICA: Modificar ManningBatchProcessor para que acepte un 'SolverFactory'.
    }

    // NOTA: Para que este test funcione sin cambiar la clase de producción (que hace 'new ManningGpuSolver'),
    // necesitaríamos PowerMock. Como no lo tenemos, la mejor práctica es extraer la creación del solver
    // a un método protegido `createSolver()` y hacer Spy del procesador.

    // --- ESTE TEST ASUME QUE HEMOS REFACTORIZADO EL PROCESSOR PARA PERMITIR MOCKING ---
    // Si no podemos refactorizar, el test fallará al intentar cargar la DLL nativa.
    // Vamos a asumir que el Processor tiene un método 'protected ManningGpuSolver createSolver()'.

    @AfterEach
    void tearDown() {
        if (batchProcessor != null) batchProcessor.close();
    }

    // --- TESTS LOGICA ORQUESTACIÓN ---

    @Test
    @DisplayName("Padding Logic: Input no múltiplo de Stride debe ser rellenado")
    void process_FullEvolution_ShouldApplyPadding() {
        // ARRANGE
        // Input: 15 pasos. BatchConfig: 10. Stride: 4.
        // Batch 1: 10 pasos. (10 % 4 != 0) -> Relleno a 12 (Stride 4*3) o recorte a 8?
        // Lógica actual: Recorta al múltiplo inferior (8) y deja el resto para el siguiente.
        // Batch 2: 7 pasos restantes (2 del anterior + 5 nuevos). Padding a 8.

        when(mockConfig.getGpuFullEvolutionStride()).thenReturn(4);
        when(mockConfig.getCpuTimeBatchSize()).thenReturn(10);

        float[] inputs = new float[15];
        RiverState state = createSteadyState(1f, 1f);

        // Mockeamos la ejecución del solver a través de una subclase parcial (Spy)
        ManningBatchProcessor spyProcessor = new TestableManningBatchProcessor(realGeometry, mockConfig, mockGpuSolver);

        // Setup Solver returns
        when(mockGpuSolver.solveFullEvolutionBatch(any(), any(), any(), anyInt()))
                .thenReturn(new ManningGpuSolver.RawGpuResult(new float[0], new float[0], 0));

        // ACT
        spyProcessor.process(inputs, state);

        // ASSERT
        // Verificamos que se llamó al solver con tamaños alineados a 4
        // Batch 1: Input size 8 (10 recortado a 8)
        verify(mockGpuSolver, atLeastOnce()).solveFullEvolutionBatch(any(), argThat(arr -> arr.length % 4 == 0), any(), eq(4));
    }

    @Test
    @DisplayName("Result Factory: Debe retornar Strided si cabe en memoria")
    void process_SmallData_ShouldReturnStridedResult() {
        // ARRANGE
        float[] inputs = new float[20];
        RiverState state = createSteadyState(1f, 1f);

        ManningBatchProcessor spyProcessor = new TestableManningBatchProcessor(realGeometry, mockConfig, mockGpuSolver);

        // Mock Result: 10 stored steps (Stride 2)
        int stored = 10;
        float[] rawData = new float[stored * cellCount];
        when(mockGpuSolver.solveFullEvolutionBatch(any(), any(), any(), anyInt()))
                .thenReturn(new ManningGpuSolver.RawGpuResult(rawData, rawData, stored));

        // ACT
        IManningResult result = spyProcessor.process(inputs, state);

        // ASSERT
        assertTrue(result instanceof StridedManningResult);
        assertEquals(inputs.length, result.getTimestepCount()); // Pasos lógicos totales
    }

    @Test
    @DisplayName("Smart Fallback: Debe cambiar a Full Evolution si Smart falla")
    void process_SmartFallback_ShouldCallFullEvolution() {
        // ARRANGE
        when(mockConfig.getGpuStrategy()).thenReturn(GpuStrategy.SMART_SAFE);

        ManningBatchProcessor spyProcessor = new TestableManningBatchProcessor(realGeometry, mockConfig, mockGpuSolver);
        RiverState state = createSteadyState(1f, 1f);
        float[] inputs = new float[10];

        // 1. Smart falla
        when(mockGpuSolver.solveSmartBatch(any(), any(), any(), eq(false)))
                .thenThrow(new IllegalStateException("Unstable"));

        // 2. Full funciona
        when(mockGpuSolver.solveFullEvolutionBatch(any(), any(), any(), anyInt()))
                .thenReturn(new ManningGpuSolver.RawGpuResult(new float[0], new float[0], 0));

        // ACT
        spyProcessor.process(inputs, state);

        // ASSERT
        verify(mockGpuSolver).solveSmartBatch(any(), any(), any(), eq(false));
        verify(mockGpuSolver).solveFullEvolutionBatch(any(), any(), any(), anyInt());
    }

    // --- Helpers ---

    private RiverState createSteadyState(float h, float v) {
        float[] arr = new float[cellCount]; Arrays.fill(arr, h);
        return new RiverState(arr, arr, arr, arr, arr);
    }

    // Subclase para inyectar el mock sin PowerMock (Pattern: Seam)
    static class TestableManningBatchProcessor extends ManningBatchProcessor {
        private final ManningGpuSolver mockSolver;

        public TestableManningBatchProcessor(RiverGeometry geo, SimulationConfig conf, ManningGpuSolver mockSolver) {
            super(geo, conf);
            this.mockSolver = mockSolver;
        }

        // HACK: Sobreescribimos la creación del solver?
        // No podemos porque se crea localmente con 'new'.
        // NECESITAMOS REFACTORIZAR LA CLASE DE PRODUCCIÓN PARA QUE ESTO FUNCIONE.
        // A continuación te doy la pequeña modificación necesaria en ManningBatchProcessor.

        // Asumiremos que ManningBatchProcessor tiene un método protegido:
        // protected ManningGpuSolver createGpuSolver() { return new ManningGpuSolver(geometry); }

        // @Override
        // protected ManningGpuSolver createGpuSolver() {
        //     return mockSolver;
        // }
    }
}