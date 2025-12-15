package projectstalker.ui.service;

import javafx.animation.AnimationTimer;
import lombok.Getter;
import lombok.Setter;
import org.springframework.stereotype.Service;
import projectstalker.config.RiverConfig;
import projectstalker.domain.river.HydrologySnapshot;
import projectstalker.domain.river.RiverGeometry;

// Importamos tus modelos y decoradores del dominio
import projectstalker.physics.model.*;

import java.util.function.Consumer;

@Service
public class SimulationEngine {

    // --- Estado de la Simulación ---
    @Getter
    private double simTimeSeconds = 0.0;
    @Getter
    private double playbackSpeed = 0.0;

    // --- Control de Pasos (Step Logic) ---
    private double lastSystemTimeNano = -1;
    private double realTimeAccumulator = 0.0;

    private static final double SIMULATION_STEP_SECONDS = 3660.0; // 1 hora por tick
    private static final double BASE_REAL_INTERVAL_SECONDS = 1.0; // 1 seg real = 1 tick

    private boolean forceUpdate = false;

    // --- ARQUITECTURA: Modelos de Dominio (Interfaces) ---
    // El Engine no sabe cómo se calculan, solo sabe que evolucionan en el tiempo.
    private TimeEvolutionModel tempModel;
    private TimeEvolutionModel phModel;
    private TimeEvolutionModel decayModel;

    @Setter
    private Consumer<HydrologySnapshot> onFrameReady;

    private final AnimationTimer timer = new AnimationTimer() {
        @Override
        public void handle(long now) {
            if (lastSystemTimeNano < 0) {
                lastSystemTimeNano = now;
                return;
            }

            double dtRealSeconds = (now - lastSystemTimeNano) / 1_000_000_000.0;
            lastSystemTimeNano = now;

            boolean needRender = false;

            if (playbackSpeed != 0.0) {
                realTimeAccumulator += dtRealSeconds;
                double targetInterval = BASE_REAL_INTERVAL_SECONDS / Math.abs(playbackSpeed);

                if (realTimeAccumulator >= targetInterval) {
                    while (realTimeAccumulator >= targetInterval) {
                        realTimeAccumulator -= targetInterval;
                        simTimeSeconds += (playbackSpeed > 0) ? SIMULATION_STEP_SECONDS : -SIMULATION_STEP_SECONDS;
                    }
                    if (simTimeSeconds < 0) {
                        simTimeSeconds = 0;
                        playbackSpeed = 0;
                        realTimeAccumulator = 0;
                    }
                    needRender = true;
                }
            } else {
                realTimeAccumulator = 0.0;
            }

            // Ejecutamos solo si tenemos los modelos instanciados
            if (tempModel != null && onFrameReady != null) {
                if (needRender || forceUpdate) {
                    calculateFrame();
                    forceUpdate = false;
                }
            }
        }
    };

    /**
     * AQUÍ ES DONDE BRILLA TU ARQUITECTURA.
     * Ensamble de la cadena de responsabilidad / Decoradores.
     */
    public void loadRiver(RiverGeometry geo, RiverConfig conf) {
        // 1. Construcción de la Cadena de Temperatura (Base + Decoradores)
        // Base: Clima (Estacional/Diario)
        TemperatureModel tempChain = new RiverTemperatureModel(conf, geo);

        // Decorador: Enfriamiento de Cabecera (Resuelve el problema de la línea recta al inicio)
        tempChain = new HeadwaterCoolingDecorator(tempChain, conf, geo);

        // Decorador: Efectos Geomorfológicos (Ancho/Pendiente)
        tempChain = new GeomorphologyTemperatureDecorator(tempChain, conf, geo);

        // Decorador: Estocástico (Ruido Perlin para realismo local)
        tempChain = new StochasticTemperatureDecorator(tempChain, conf);

        this.tempModel = tempChain;

        // 2. Construcción del Modelo de pH
        this.phModel = new RiverPhModel(conf, geo, tempChain);

        // 3. Construcción del Modelo de Decay
        // El Decay necesita la Temperatura para la corrección de Arrhenius.
        // Inyectamos el modelo de temperatura ya decorado.
        this.decayModel = new RiverDecayModel(geo, tempChain);

        // Reset estado
        this.simTimeSeconds = 0;
        this.playbackSpeed = 0;
        this.realTimeAccumulator = 0;
        this.forceUpdate = true;
    }

    public void setPlaybackSpeed(double speed) {
        this.playbackSpeed = speed;
        this.realTimeAccumulator = 0;
        this.forceUpdate = true;
    }

    public void start() {
        timer.start();
        lastSystemTimeNano = -1;
    }

    public void stop() {
        timer.stop();
    }

    public void restartTime() {
        this.simTimeSeconds = 0.0;
        this.playbackSpeed = 0.0;
        this.realTimeAccumulator = 0;
        this.forceUpdate = true;
    }

    // --- Núcleo Matemático ---

    private void calculateFrame() {
        // DELEGACIÓN PURA:
        // El Engine no calcula nada. Pide los datos a los modelos expertos.

        // 1. Temperatura (Llama a toda la cadena de decoradores)
        float[] tempOut = tempModel.generateProfile(simTimeSeconds);

        // 2. pH
        float[] phOut = phModel.generateProfile(simTimeSeconds);

        // 3. Decay (Internamente usará tempModel.generateProfile o similar para Arrhenius)
        float[] decayOut = decayModel.generateProfile(simTimeSeconds);

        // Empaquetar y enviar
        HydrologySnapshot snapshot = new HydrologySnapshot(tempOut, phOut, decayOut, simTimeSeconds);
        onFrameReady.accept(snapshot);
    }
}