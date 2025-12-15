package projectstalker.ui.service;

import javafx.animation.AnimationTimer;
import lombok.Getter;
import lombok.Setter;
import org.springframework.stereotype.Service;
import projectstalker.config.RiverConfig;
import projectstalker.domain.river.HydrologySnapshot;
import projectstalker.domain.river.RiverGeometry;

import java.util.function.Consumer;
import java.util.stream.IntStream;

/**
 * Motor de Simulación Hidrológica en Tiempo Real.
 * <p>
 * Responsabilidades:
 * 1. Gestionar el avance del tiempo (Play, Pause, Rewind, Speed).
 * 2. Orquestar el cálculo paralelo de modelos dinámicos (Temp, pH, Decay).
 * 3. Emitir snapshots sincronizados con el framerate de la UI (60 FPS).
 */
@Service
public class SimulationEngine {

    // --- Estado de la Simulación ---
    @Getter
    private double simTimeSeconds = 0.0;
    @Getter
    @Setter
    private double playbackSpeed = 0.0; // 0.0 = Pausa
    private double lastSystemTimeNano = -1;

    // --- Estado del Río (Cache para rendimiento) ---
    private RiverGeometry geometry;
    private RiverConfig config;
    private float[] basePhProfile;   // Perfil espacial base (Geología)
    private float[] baseDecayProfile; // Perfil espacial base (Manning)

    /**
     * -- SETTER --
     * Configura el callback para pintar.
     */
    // --- Callback UI ---
    @Setter
    private Consumer<HydrologySnapshot> onFrameReady;

    // --- Constantes Físicas ---
    private static final double SECONDS_PER_DAY = 86400.0;
    private static final double TAU = 2.0 * Math.PI;

    // --- El Bucle Principal ---
    private final AnimationTimer timer = new AnimationTimer() {
        @Override
        public void handle(long now) {
            if (lastSystemTimeNano < 0) {
                lastSystemTimeNano = now;
                return;
            }

            // 1. Calcular Delta Time Real (dt)
            double dtRealSeconds = (now - lastSystemTimeNano) / 1_000_000_000.0;
            lastSystemTimeNano = now;

            // 2. Avanzar Tiempo de Simulación
            if (playbackSpeed != 0.0) {
                simTimeSeconds += dtRealSeconds * playbackSpeed;

                // Clamp: No permitir tiempo negativo
                if (simTimeSeconds < 0) {
                    simTimeSeconds = 0;
                    playbackSpeed = 0; // Auto-pausa al llegar al inicio
                }
            }

            // 3. Ejecutar Física Solo si tenemos datos cargados
            if (geometry != null && config != null && onFrameReady != null) {
                calculateFrame();
            }
        }
    };

    /**
     * Carga un nuevo río en el motor y pre-calcula los datos estáticos.
     */
    public void loadRiver(RiverGeometry geo, RiverConfig conf) {
        this.geometry = geo;
        this.config = conf;

        // Copias defensivas de los perfiles base espaciales
        // (El motor aplicará variaciones temporales sobre esto)
        this.basePhProfile = geo.clonePhProfile();
        this.baseDecayProfile = geo.cloneBaseDecayCoefficientAt20C();

        // Resetear tiempo al cargar nuevo río (opcional)
        this.simTimeSeconds = 0;
        this.playbackSpeed = 0; // Pausa por defecto
    }

    // --- Controles de Reproducción ---

    public void start() {
        timer.start();
        lastSystemTimeNano = -1;
    }

    public void stop() {
        timer.stop();
    }

    public void restartTime() {
        this.simTimeSeconds = 0.0;
        this.playbackSpeed = 0.0; // Pausar al reiniciar
    }

    public double getCurrentSimTime() {
        return simTimeSeconds;
    }

    // --- NÚCLEO MATEMÁTICO PARALELO ---

    private void calculateFrame() {
        int count = geometry.getCellCount();

        // Arrays de salida para este frame
        float[] tempOut = new float[count];
        float[] phOut = new float[count];
        float[] decayOut = new float[count];

        // --- 1. PRE-CÁLCULO TEMPORAL (Optimizacion: Fuera del bucle) ---
        // Estos valores son constantes para todo el río en este instante 't'.
        double t = simTimeSeconds;

        // A. Ciclos de Tiempo
        double dayCycle = (t % SECONDS_PER_DAY) / SECONDS_PER_DAY * TAU;
        double secondsPerYear = SECONDS_PER_DAY * 365.0;
        double yearCycle = (t % secondsPerYear) / secondsPerYear * TAU;

        // B. Factor Estacional (-1.0 Invierno a 1.0 Verano)
        // Empezamos en -PI/2 para que t=0 sea Invierno (mínimo)
        double seasonalFactor = Math.sin(yearCycle - (Math.PI / 2.0));
        double currentSeasonalBase = config.averageAnualTemperature()
                + (config.seasonalTempVariation() * seasonalFactor);

        // C. Factor Diario (-1.0 Noche a 1.0 Tarde)
        // Pico a las 15:00h
        double dailyPhaseShift = ((15.0 / 24.0) * TAU) - (Math.PI / 2.0);
        double dailyFactor = Math.sin(dayCycle - dailyPhaseShift);

        // D. Temperatura Global del Agua (Sin efectos locales aún)
        float globalTemp = (float) (currentSeasonalBase + (config.dailyTempVariation() * dailyFactor));
        globalTemp = Math.max(0.1f, globalTemp); // El agua no se congela en esta simulación

        // E. Factor Biológico pH (Retraso respecto al sol)
        double phPhase = ((config.riverPhaseShiftHours() + 1.0) / 24.0) * TAU;
        double bioFactor = Math.sin(dayCycle - phPhase);
        float phVariation = (float) (config.phVariability() * 0.5 * bioFactor);

        // Constantes para Decay
        final float theta = 1.047f; // Coeficiente Arrhenius estándar

        // --- 2. EJECUCIÓN PARALELA (Cálculo Espacial) ---
        // Aplicamos los factores temporales a cada celda espacial específica.
        final float finalGlobalTemp = globalTemp; // Efectivamente final para lambda

        IntStream.range(0, count).parallel().forEach(i -> {

            // A. Temperatura (Aquí podrías sumar enfriamiento de cabecera si quisieras espacialidad)
            tempOut[i] = finalGlobalTemp;

            // B. pH (Base Geológica Local + Variación Biológica Temporal)
            phOut[i] = basePhProfile[i] + phVariation;

            // C. Decay (Base Local k20 corregida por Temperatura Global)
            // k_T = k_20 * theta^(T - 20)
            float k20 = baseDecayProfile[i];
            double temperatureCorrection = Math.pow(theta, finalGlobalTemp - 20.0);
            decayOut[i] = (float) (k20 * temperatureCorrection);
        });

        // --- 3. PUBLICAR RESULTADOS ---
        HydrologySnapshot snapshot = new HydrologySnapshot(
                tempOut, phOut, decayOut, t
        );

        onFrameReady.accept(snapshot);
    }
}