package projectstalker.ui.renderer;

import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;
import projectstalker.config.RiverConfig;
import projectstalker.utils.FastNoiseLite;

public class NoiseSignatureRenderer {
    public static void render(Canvas noiseCanvas, RiverConfig config){
        GraphicsContext gc = noiseCanvas.getGraphicsContext2D();
        double w = noiseCanvas.getWidth();
        double h = noiseCanvas.getHeight();

        // Limpiar fondo
        gc.setFill(Color.web("#2E3440"));
        gc.fillRect(0, 0, w, h);

        // 1. Obtención de Parámetros
        long seed = config.seed();
        double mainFreq = config.noiseFrequency();
        double detailFreq = config.detailNoiseFrequency();
        double zoneFreq = config.zoneNoiseFrequency();

        // 2. Inicializar los Generadores de Ruido

        // Generador PRINCIPAL (usa la frecuencia del Slider)
        final FastNoiseLite mainNoise = new FastNoiseLite((int) seed); // Usa la semilla base
        mainNoise.SetNoiseType(FastNoiseLite.NoiseType.Perlin);
        mainNoise.SetFrequency((float) mainFreq);

        // Generador de Detalle
        final FastNoiseLite detailNoise = new FastNoiseLite((int) seed + 1);
        detailNoise.SetNoiseType(FastNoiseLite.NoiseType.Perlin);
        detailNoise.SetFrequency((float) detailFreq);

        // Generador de Zona
        final FastNoiseLite zoneNoise = new FastNoiseLite((int) seed + 2);
        zoneNoise.SetNoiseType(FastNoiseLite.NoiseType.Perlin);
        zoneNoise.SetFrequency((float) zoneFreq);

        // 3. Configuración de Dibujo
        gc.setStroke(Color.web("#88C0D0"));
        gc.setLineWidth(2.0);
        gc.beginPath();

        double yBase = h / 2;
        double maxAmplitude = h * 0.45;
        double scaleX = 0.5;

        for (int x = 0; x < w; x++) {
            double worldX = x * scaleX;
            int i = (int) worldX;

            // Obtener los valores de ruido normalizados a [-1, 1]
            double currentMainNoise = mainNoise.GetNoise(i, 0);
            double currentDetailNoise = detailNoise.GetNoise(i, 0);
            double currentZoneNoise = zoneNoise.GetNoise(i, 0);

            // Generar la forma de onda combinada (Ponderación)
            // Damos más peso al ruido principal y zonal, y menos al detalle.
            // La suma debe ser aproximada a 1.0 para mantener la amplitud.
            double noiseEffect = (currentMainNoise * 0.6) +  // Principal (Slider)
                    (currentZoneNoise * 0.3) +  // Zonal (Frecuencia Macro)
                    (currentDetailNoise * 0.1); // Detalle (Frecuencia Fina)

            // Normalizar (ya que noiseEffect ahora está escalado)
            double yFinal = yBase + (noiseEffect * maxAmplitude);

            if (x == 0) gc.moveTo(x, yFinal);
            else gc.lineTo(x, yFinal);
        }
        gc.stroke();

        // Overlay de texto
        gc.setFill(Color.WHITE);
        gc.fillText("Firma de Ruido (Seed: " + seed + ")", 10, 20);
    }
}
