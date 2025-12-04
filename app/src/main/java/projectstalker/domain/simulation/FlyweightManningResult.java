package projectstalker.domain.simulation;

import lombok.Builder;
import lombok.Getter;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.domain.river.RiverState;

import java.util.Arrays;

/**
 * Implementación "Flyweight" (Lazy) de los resultados de la simulación.
 * <p>
 * Diseñada para el solver de GPU. En lugar de almacenar una matriz densa de todo el río
 * para cada paso de tiempo (lo que saturaría la RAM), almacena solo:
 * <ol>
 * <li>El estado inicial (Semilla / Intrinsic State).</li>
 * <li>Los datos nuevos calculados por la GPU (Triángulo activo / Extrinsic State).</li>
 * </ol>
 * <p>
 * Cuando se solicita un estado mediante {@link #getStateAt(int)}, este se reconstruye al vuelo
 * combinando ambas fuentes de datos, simulando el desplazamiento de la ola.
 */
@Builder
public class FlyweightManningResult implements ISimulationResult {

    @Getter
    private final RiverGeometry geometry;

    @Getter
    private final long simulationTime;

    // --- ESTADO INTRÍNSECO (Compartido / Base) ---
    // Guardamos arrays primitivos para acceso rápido sin overhead de objetos
    private final float[] initialDepths;
    private final float[] initialVelocities;

    // --- ESTADO EXTRÍNSECO (Deltas de GPU) ---
    // Matriz compacta [Time][Var][ActiveWidth] devuelta por el Solver JNI
    // ActiveWidth suele ser == Time (BatchSize), formando un triángulo de datos nuevos.
    private final float[][][] gpuPackedData;

    // Datos auxiliares (Temperatura/pH) que no se calculan en este kernel de GPU
    // Se mantienen en memoria (son ligeros comparados con la hidráulica si no varían mucho)
    private final float[][][] auxData;

    /**
     * Constructor personalizado para extraer arrays crudos del estado inicial
     * y evitar llamadas repetidas a getters que podrían clonar.
     */
    public FlyweightManningResult(RiverGeometry geometry,
                                  long simulationTime,
                                  RiverState initialState,
                                  float[][][] gpuPackedData,
                                  float[][][] auxData) {
        this.geometry = geometry;
        this.simulationTime = simulationTime;

        // Extracción defensiva inicial (Flyweight Base)
        this.initialDepths = initialState.waterDepth();
        this.initialVelocities = initialState.velocity();

        this.gpuPackedData = gpuPackedData;
        this.auxData = auxData;
    }

    // Constructor generado por Lombok (@Builder) necesita un adaptador manual
    // si queremos la lógica de extracción arriba. Para simplificar con Lombok,
    // usaremos un factory method o un constructor manual limpio.
    // Aquí, usamos el constructor manual estándar compatible con el Builder.
    public static class FlyweightManningResultBuilder {
        // Personalizamos el build() si fuera necesario, pero por ahora Lombok lo maneja.
    }

    @Override
    public int getTimestepCount() {
        return gpuPackedData.length;
    }

    /**
     * Reconstruye el estado del río en el tiempo 't' bajo demanda.
     * <p>
     * Algoritmo de Reconstrucción (Smart Fetch Reconstruction):
     * 1. Zona Nueva (x <= t): Copia directa desde gpuPackedData.
     * 2. Zona Antigua (x > t): Copia desde initialDepths desplazado (Shift).
     * Lógica: initial[x - (t + 1)]
     */
    @Override
    public RiverState getStateAt(int t) {
        int cellCount = geometry.getCellCount();

        // Arrays destino (Se crean al vuelo -> Pasan al GC rápido tras usarse)
        float[] h = new float[cellCount];
        float[] v = new float[cellCount];

        // ------------------------------------------------------------
        // FASE 1: ZONA NUEVA (Datos de GPU)
        // ------------------------------------------------------------
        // El array de GPU tiene tamaño 'activeWidth', que idealmente es 'batchSize'.
        // Copiamos todo lo que la GPU nos dio para este paso 't'.
        float[] gpuH = gpuPackedData[t][0];
        float[] gpuV = gpuPackedData[t][1];

        // La cantidad de datos nuevos válidos es (t + 1), limitado por el ancho del buffer GPU
        // y el tamaño del río.
        int newDataLimit = Math.min(t + 1, gpuH.length);

        // Si el río es muy corto, no nos salimos.
        int copyLength = Math.min(newDataLimit, cellCount);

        // Copia masiva de memoria (Muy rápido)
        System.arraycopy(gpuH, 0, h, 0, copyLength);
        System.arraycopy(gpuV, 0, v, 0, copyLength);

        // ------------------------------------------------------------
        // FASE 2: ZONA ANTIGUA (Estado Inicial Desplazado)
        // ------------------------------------------------------------
        // Rellenamos el resto del río con el agua que "ya estaba ahí" y ha fluido hacia abajo.
        // Índice destino empieza donde terminó la GPU: 'copyLength'.
        // Índice origen: El agua que ahora está en 'copyLength' estaba en '0' hace 'copyLength' tiempo.
        // Fórmula del Kernel: src = dst - (t + 1).
        // Si dst = t + 1 -> src = 0. Correcto.

        int remainingCells = cellCount - copyLength;

        if (remainingCells > 0) {
            // Origen en el array inicial: 0 (el principio del río viejo ahora está aquí)
            // Destino en el array nuevo: copyLength
            // Cantidad: Lo que quepa

            // Validación de seguridad para no leer fuera del array inicial
            // (Si el batch es mayor que el río, remainingCells será 0 y no entramos aquí)
            System.arraycopy(initialDepths, 0, h, copyLength, remainingCells);
            System.arraycopy(initialVelocities, 0, v, copyLength, remainingCells);
        }

        // ------------------------------------------------------------
        // FASE 3: CONSTRUCCIÓN DEL ESTADO
        // ------------------------------------------------------------
        // Recuperamos datos auxiliares si existen, o ceros.
        float t_val = (auxData != null && t < auxData.length) ? auxData[t][0][0] : 0f; // Simplificación temporal
        float ph_val = (auxData != null && t < auxData.length) ? auxData[t][1][0] : 0f;

        // Construcción eficiente usando arrays primitivos pre-sanitizados
        // Nota: Asumimos temperatura y pH constantes espacialmente por ahora en este modelo flyweight,
        // o arrays llenos si 'auxData' lo soporta. Para mantenerlo ligero, aquí llenamos arrays
        // temporales o usamos un builder inteligente.

        float[] tempArr = new float[cellCount];
        float[] phArr = new float[cellCount];
        Arrays.fill(tempArr, t_val);
        Arrays.fill(phArr, ph_val);

        return RiverState.builder()
                .waterDepth(h)
                .velocity(v)
                .temperature(tempArr)
                .ph(phArr)
                .contaminantConcentration(new float[cellCount]) // Transporte va aparte
                .build();
    }
}