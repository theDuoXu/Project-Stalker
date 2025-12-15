package projectstalker.ui.view.util;

import projectstalker.config.RiverConfig;

public class RiverPresets {

    public static RiverConfig standard() {
        return RiverConfig.getTestingRiver(); // Tu default del backend
    }

    public static RiverConfig mountainTorrent() {
        return standard()
                .withBaseWidth(25.0f)           // Estrecho
                .withAverageSlope(0.015f)       // Muy empinado (1.5%)
                .withBaseManning(0.050f)        // Muy rugoso (rocas)
                .withConcavityFactor(0.8f)      // Perfil muy cóncavo
                .withNoiseFrequency(0.02f)      // Muy caótico
                .withTemperatureNoiseAmplitude(1.5f); // Cambios bruscos de temp
    }

    public static RiverConfig widePlains() {
        return standard()
                .withBaseWidth(300.0f)          // Muy ancho
                .withAverageSlope(0.00005f)     // Casi plano
                .withBaseManning(0.025f)        // Fondo arenoso/suave
                .withConcavityFactor(0.2f)      // Perfil suave
                .withNoiseFrequency(0.001f);    // Meandros suaves
    }
}