package projectstalker.ui.renderer;

import javafx.application.Platform;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Label;
import javafx.scene.paint.Color;
import javafx.scene.paint.CycleMethod;
import javafx.scene.paint.LinearGradient;
import javafx.scene.paint.Stop;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;
import javafx.scene.Node;
import lombok.Getter;
import projectstalker.domain.river.RiverGeometry;

/**
 * Renderizador del río con soporte para HUD flotante y Modos de Visualización.
 */
public class RiverRenderer {

    private final Canvas canvas;
    private ThemePalette palette;

    // Record interno para agrupar semánticamente los colores
    private record ThemePalette(
            Color bg, Color text, Color border,
            Color waterDeep, Color waterSurface,
            Color sand, Color rock,
            Color acidZone, Color baseZone, // Ahora son colores base brillantes
            Color cursor,
            Color analysisLine1, Color analysisLine2
    ) {
    }

    @Getter
    public enum RenderMode {
        MORPHOLOGY(false, "Vista realista (Morfología)"),
        ANALYSIS(true, "Vista de datos (Hidráulica y Química)");

        private final boolean isAnalyticalMode;
        private final String title;

        RenderMode(boolean isAnalyticalMode, String title) {
            this.isAnalyticalMode = isAnalyticalMode;
            this.title = title;
        }

        public static RenderMode fromBoolean(boolean isAnalyticalMode) {
            if (isAnalyticalMode) return ANALYSIS;
            return MORPHOLOGY;
        }
    }

    public RiverRenderer(Canvas canvas) {
        this.canvas = canvas;
        this.palette = createFallbackPalette();
    }

    /**
     * Fuerza la recarga de los colores desde el CSS actual.
     */
    public void reloadThemeColors() {
        if (canvas.getScene() == null) return;
        Node root = canvas.getScene().getRoot();

        this.palette = new ThemePalette(
                resolveColor(root, "-color-bg-default", Color.web("#2E3440")),
                resolveColor(root, "-color-fg-default", Color.WHITE),
                resolveColor(root, "-color-border-muted", Color.LIGHTGRAY),

                resolveColor(root, "-color-accent-emphasis", Color.BLUE),
                resolveColor(root, "-color-accent-subtle", Color.LIGHTBLUE),

                resolveColor(root, "-color-warning-emphasis", Color.SANDYBROWN),
                resolveColor(root, "-color-neutral-emphasis-plus", Color.DARKGRAY),

                // Usamos colores de "Danger" y "Success" pero aseguramos opacidad para las gráficas
                resolveColor(root, "-color-danger-fg", Color.web("#FF5555")), // Rojo brillante
                resolveColor(root, "-color-success-fg", Color.web("#50FA7B")), // Verde brillante

                resolveColor(root, "-color-fg-default", Color.WHITE).deriveColor(0, 1, 1, 0.8),

                // Colores extra para gráficas (Geometry)
                Color.web("#F8F8F2"), // Blanco (Ancho)
                Color.web("#BD93F9")  // Morado (Talud)
        );
    }

    private ThemePalette createFallbackPalette() {
        return new ThemePalette(
                Color.web("#2E3440"), Color.WHITE, Color.GRAY,
                Color.BLUE, Color.LIGHTBLUE, Color.SANDYBROWN, Color.DARKGRAY,
                Color.RED, Color.GREEN, Color.WHITE,
                Color.WHITE, Color.MAGENTA
        );
    }

    private Color resolveColor(Node root, String cssVariable, Color fallback) {
        try {
            Label dummy = new javafx.scene.control.Label();
            dummy.setVisible(false);
            dummy.setManaged(false);
            javafx.scene.layout.Pane container = null;

            if (root instanceof javafx.scene.layout.Pane) container = (javafx.scene.layout.Pane) root;
            else if (canvas.getParent() instanceof javafx.scene.layout.Pane)
                container = (javafx.scene.layout.Pane) canvas.getParent();

            if (container == null) return fallback;

            container.getChildren().add(dummy);
            dummy.setStyle("-fx-text-fill: " + cssVariable + ";");
            dummy.applyCss();
            javafx.scene.paint.Paint paint = dummy.getTextFill();
            Color result = (paint instanceof Color) ? (Color) paint : fallback;
            container.getChildren().remove(dummy);
            return result;
        } catch (Exception e) {
            return fallback;
        }
    }

    /**
     * Renderiza la escena.
     *
     * @param onRenderFinished Callback opcional (Runnable) para lanzar eventos al terminar.
     */
    public void render(RiverGeometry geo, RenderMode mode, double mouseX, double mouseY, Runnable onRenderFinished) {

        Platform.runLater(() -> {
            GraphicsContext gc = canvas.getGraphicsContext2D();
            double w = canvas.getWidth();
            double h = canvas.getHeight();

            gc.clearRect(0, 0, w, h);
            gc.setFill(palette.bg());
            gc.fillRect(0, 0, w, h);

            if (geo != null) {
                if (mode == RenderMode.MORPHOLOGY) {
                    drawPlanView(gc, geo, w, h);
                } else {
                    drawAnalysisView(gc, geo, w, h);
                }

                if (mouseX >= 0 && mouseX < w && mouseY >= 0 && mouseY < h) {
                    drawFloatingHUD(gc, geo, mode, mouseX, mouseY, w, h);
                }
            }

            if (onRenderFinished != null) {
                onRenderFinished.run();
            }
        });
    }

    // Sobrecarga simple sin callback para eventos de ratón (performance)
    public void render(RiverGeometry geo, RenderMode mode, double mouseX, double mouseY) {
        render(geo, mode, mouseX, mouseY, null);
    }

    /**
     * VISTA REALISTA: Solo Agua y Material (Arena/Roca). Limpia.
     */
    private void drawPlanView(GraphicsContext gc, RiverGeometry geo, double w, double h) {
        float[] widths = geo.getBottomWidth();
        int cells = geo.getCellCount();

        double scaleX = w / cells;
        double centerY = h / 2.0;

        // --- CORRECCIÓN DE ESCALA ---
        double maxRiverWidth = 0;
        for (float width : widths) maxRiverWidth = Math.max(maxRiverWidth, width);

        // Definimos un Ancho de Cámara mínimo de 250 metros.
        // - Si el río mide 10m, el divisor es 250m -> ScaleY bajo -> El río se ve fino.
        // - Si el río mide 500m, el divisor es 500m -> ScaleY se ajusta -> El río cabe justo.
        double viewportMeters = Math.max(maxRiverWidth, 250.0);

        double scaleY = (h * 0.9) / viewportMeters;
        // ---------------------------

        for (int i = 0; i < cells; i++) {
            double x = i * scaleX;
            double cellW = Math.max(1.0, scaleX + 0.6);

            double riverWidthPx = widths[i] * scaleY;
            double topY = centerY - (riverWidthPx / 2.0);

            gc.setFill(palette.waterDeep());
            gc.fillRect(x, topY, cellW, riverWidthPx);

        }
        drawMorphologyStatsOverlay(gc, geo);
    }

    /**
     * VISTA ANALÍTICA: 5 Gráficas de líneas normalizadas + Sombra de contexto.
     */
    private void drawAnalysisView(GraphicsContext gc, RiverGeometry geo, double w, double h) {
        int cells = geo.getCellCount();

        // Datos
        float[] mannings = geo.getManningCoefficient();
        float[] phs = geo.getPhProfile();
        float[] decays = geo.getBaseDecayCoefficientAt20C();
        float[] widths = geo.getBottomWidth();
        float[] slopes = geo.getSideSlope();

        double scaleX = w / cells;
        double centerY = h / 2.0;

        // 1. SILUETA FANTASMA (Contexto)
        // Aplicamos la misma corrección aquí para que la sombra coincida con la realidad visual
        double maxRiverWidth = 0;
        for (float width : widths) maxRiverWidth = Math.max(maxRiverWidth, width);

        // Usar el mismo Viewport Mínimo
        double viewportMeters = Math.max(maxRiverWidth, 250.0);
        double scaleYRiver = (h * 0.9) / viewportMeters;

        gc.setFill(palette.waterSurface().deriveColor(0, 1, 1, 0.1));

        for (int i = 0; i < cells; i++) {
            double x = i * scaleX;
            double riverW = widths[i] * scaleYRiver;
            gc.fillRect(x, centerY - (riverW / 2), Math.max(1.0, scaleX), riverW);
        }

        // 2. PREPARACIÓN DE GRÁFICAS
        Color colWidth = palette.analysisLine1();
        Color colSlope = palette.analysisLine2();
        Color colManning = palette.sand();
        Color colPh = palette.acidZone();
        Color colDecay = palette.waterDeep().brighter();

        double[] xP = new double[cells];
        double[] yWidth = new double[cells];
        double[] ySlope = new double[cells];
        double[] yManning = new double[cells];
        double[] yPh = new double[cells];
        double[] yDecay = new double[cells];

        double margin = 30;
        double gH = h - (margin * 2);

        for (int i = 0; i < cells; i++) {
            xP[i] = i * scaleX;

            // Width: Aquí normalizamos contra su propio máximo para ver la FORMA de la variación,
            // independientemente de si el río es pequeño o grande.
            // Si quieres ver la magnitud absoluta, usa viewportMeters en lugar de maxRiverWidth.
            yWidth[i] = (h - margin) - ((widths[i] / (maxRiverWidth * 1.1)) * gH);

            ySlope[i] = (h - margin) - ((slopes[i] / 5.0) * gH);
            double normM = (mannings[i] - 0.02) / 0.04;
            yManning[i] = (h - margin) - (Math.max(0, Math.min(1, normM)) * gH);
            double normPh = (phs[i] - 6.0) / 3.0;
            yPh[i] = (h - margin) - (Math.max(0, Math.min(1, normPh)) * gH);
            yDecay[i] = (h - margin) - ((decays[i] / 0.5) * gH);
        }

        gc.setLineWidth(2.0);
        drawPolyline(gc, xP, yWidth, cells, colWidth);
        drawPolyline(gc, xP, ySlope, cells, colSlope);
        drawPolyline(gc, xP, yManning, cells, colManning);
        drawPolyline(gc, xP, yPh, cells, colPh);
        drawPolyline(gc, xP, yDecay, cells, colDecay);

        drawAnalysisLegend(gc, 10, 20, colWidth, colSlope, colManning, colPh, colDecay);
    }

    private void drawPolyline(GraphicsContext gc, double[] x, double[] y, int pts, Color c) {
        gc.setStroke(c);
        gc.strokePolyline(x, y, pts);
    }

    private void drawAnalysisLegend(GraphicsContext gc, double x, double y, Color... colors) {
        gc.setFill(palette.bg().deriveColor(0, 1, 1, 0.9));
        gc.setStroke(palette.border());
        gc.fillRoundRect(x, y, 140, 110, 10, 10);
        gc.strokeRoundRect(x, y, 140, 110, 10, 10);

        gc.setFont(Font.font("Arial", FontWeight.BOLD, 10));

        double ly = y + 20;
        drawLegendItem(gc, x + 10, ly, colors[0], "Ancho (b)");
        ly += 18;
        drawLegendItem(gc, x + 10, ly, colors[1], "Talud (z)");
        ly += 18;
        drawLegendItem(gc, x + 10, ly, colors[2], "Manning (n)");
        ly += 18;
        drawLegendItem(gc, x + 10, ly, colors[3], "pH (6-9)");
        ly += 18;
        drawLegendItem(gc, x + 10, ly, colors[4], "Decay (k)");
    }

    private void drawLegendItem(GraphicsContext gc, double x, double y, Color c, String text) {
        gc.setStroke(c);
        gc.setLineWidth(2.0);
        gc.strokeLine(x, y, x + 15, y);
        gc.setFill(palette.text());
        gc.fillText(text, x + 25, y + 4);
    }

    /**
     * HUD FLOTANTE
     */
    private void drawFloatingHUD(GraphicsContext gc, RiverGeometry geo, RenderMode mode, double mx, double my, double w, double h) {
        // --- A. CÁLCULOS COMUNES ---
        int cellCount = geo.getCellCount();
        double pixelsPerCell = w / cellCount;
        int index = (int) (mx / pixelsPerCell);

        if (index < 0) index = 0;
        if (index >= cellCount) index = cellCount - 1;

        // Cursor Vertical
        double cursorX = index * pixelsPerCell + (pixelsPerCell / 2.0);
        gc.setStroke(palette.cursor());
        gc.setLineWidth(1.0);
        gc.setLineDashes(4);
        gc.strokeLine(cursorX, 0, cursorX, h);
        gc.setLineDashes(0);

        // Datos Básicos
        float dist = (float) (index * geo.getSpatialResolution());
        float elev = geo.getElevationProfile()[index];

        // --- B. DIMENSIONES Y POSICIÓN ---
        // La altura de la caja varía según el modo
        double boxW = 220;
        double boxH = (mode == RenderMode.MORPHOLOGY) ? 160 : 190; // Análisis necesita más espacio vertical
        double padding = 15;

        double boxX = (mx + padding + boxW > w) ? (mx - boxW - padding) : (mx + padding);
        double boxY = (my + padding + boxH > h) ? (my - boxH - padding) : (my + padding);

        // Fondo y Borde
        gc.setFill(palette.bg().deriveColor(0, 1, 1, 0.95));
        gc.setStroke(palette.border());
        gc.setLineWidth(1.5);
        gc.fillRoundRect(boxX, boxY, boxW, boxH, 10, 10);
        gc.strokeRoundRect(boxX, boxY, boxW, boxH, 10, 10);

        // --- C. CONTENIDO COMÚN (CABECERA) ---
        gc.setFill(palette.text());
        gc.setFont(Font.font("Monospaced", FontWeight.BOLD, 12));
        double textX = boxX + 15;
        double lineH = 18;
        double curY = boxY + 25;

        // Info Geográfica básica
        gc.fillText(String.format("DIST: %.0f m", dist), textX, curY); curY += lineH;
        gc.fillText(String.format("ELEV: %.2f m", elev), textX, curY); curY += lineH + 5;

        // Separador
        gc.setStroke(Color.GRAY);
        gc.setLineWidth(0.5);
        gc.strokeLine(textX, curY - 5, boxX + boxW - 15, curY - 5);

        // --- D. CONTENIDO ESPECÍFICO POR MODO ---
        if (mode == RenderMode.MORPHOLOGY) {
            drawMorphologyHUDContent(gc, geo, index, boxX, boxW, boxY, boxH, curY, textX);
        } else {
            drawAnalysisHUDContent(gc, geo, index, textX, curY);
        }
    }

    /**
     * Contenido del HUD para el modo MORFOLOGÍA (Pendiente % + Visualización Sección)
     */
    private void drawMorphologyHUDContent(GraphicsContext gc, RiverGeometry geo, int index,
                                          double boxX, double boxW, double boxY, double boxH,
                                          double startY, double textX) {
        float bottomWidth = geo.getBottomWidth()[index];
        float z = geo.getSideSlope()[index];

        // Pendiente Local en %
        float slopePercent = calculateLocalSlopePercent(geo, index);
        gc.setFill(palette.text());
        gc.fillText(String.format("PEND: %.3f %%", slopePercent), textX, startY);

        // Mini Cross-Section
        double graphH = 50;
        double graphW = boxW - 30;
        double graphBaseY = boxY + boxH - 15;
        double graphCenterX = boxX + (boxW / 2.0);

        double visualDepth = graphH * 0.8;
        double visualBaseWidth = Math.min(graphW * 0.4, bottomWidth * 2.0);
        double visualTopWidth = visualBaseWidth + (2 * z * (visualDepth / 2.0));

        double xBL = graphCenterX - (visualBaseWidth / 2.0);
        double xBR = graphCenterX + (visualBaseWidth / 2.0);
        double xTL = graphCenterX - (visualTopWidth / 2.0);
        double xTR = graphCenterX + (visualTopWidth / 2.0);
        double yB  = graphBaseY;
        double yT  = graphBaseY - visualDepth;

        // Agua
        gc.beginPath();
        gc.moveTo(xBL, yB); gc.lineTo(xTL, yT); gc.lineTo(xTR, yT); gc.lineTo(xBR, yB);
        gc.closePath();
        LinearGradient grad = new LinearGradient(0, 0, 0, 1, true, CycleMethod.NO_CYCLE,
                new Stop(0, palette.waterSurface()), new Stop(1, palette.waterDeep));
        gc.setFill(grad);
        gc.fill();

        // Lecho
        gc.setStroke(palette.text().deriveColor(0, 1, 1, 0.5));
        gc.setLineWidth(2.0);
        gc.strokePolyline(new double[]{xTL, xBL, xBR, xTR}, new double[]{yT, yB, yB, yT}, 4);

        // Texto descriptivo
        gc.setFont(Font.font("Arial", 9));
        gc.setFill(palette.text().deriveColor(0, 1, 1, 0.7));
        String shape = (z < 1.0) ? "V (Cañón)" : "U (Valle)";
        gc.fillText(shape, xTR + 5, yT + 10);
    }

    /**
     * Contenido del HUD para el modo ANÁLISIS (Lista de valores exactos con colores)
     */
    private void drawAnalysisHUDContent(GraphicsContext gc, RiverGeometry geo, int index, double x, double startY) {
        // Datos crudos
        float w = geo.getBottomWidth()[index];
        float z = geo.getSideSlope()[index];
        float n = geo.getManningCoefficient()[index];
        float ph = geo.getPhProfile()[index];
        float k = geo.getBaseDecayCoefficientAt20C()[index];

        double y = startY;
        double lineSpacing = 18;

        // Usamos los mismos colores que las líneas para correlación visual inmediata
        drawDataRow(gc, x, y, palette.analysisLine1(), "Ancho (b):", String.format("%.2f m", w)); y += lineSpacing;
        drawDataRow(gc, x, y, palette.analysisLine2(), "Talud (z):", String.format("%.2f", z));   y += lineSpacing;
        drawDataRow(gc, x, y, palette.sand(),          "Manning (n):", String.format("%.4f", n)); y += lineSpacing;
        drawDataRow(gc, x, y, palette.acidZone(),      "pH:",          String.format("%.2f", ph));y += lineSpacing;
        drawDataRow(gc, x, y, palette.waterDeep().brighter(), "Decay (k):", String.format("%.3f", k));
    }

    private void drawDataRow(GraphicsContext gc, double x, double y, Color c, String label, String value) {
        // Marcador de color
        gc.setFill(c);
        gc.fillOval(x, y - 8, 8, 8); // Bolita de color

        // Texto Label
        gc.setFill(palette.text()); // Texto normal
        gc.setFont(Font.font("Monospaced", FontWeight.NORMAL, 12));
        gc.fillText(label, x + 15, y);

        // Texto Valor (Alineado a la derecha o seguido, aquí lo pongo seguido pero destacado)
        gc.setFont(Font.font("Monospaced", FontWeight.BOLD, 12));
        gc.fillText(value, x + 120, y); // Offset fijo para tabular valores
    }

    private float calculateLocalSlopePercent(RiverGeometry geo, int index) {
        if (index < geo.getCellCount() - 1) {
            float drop = geo.getElevationProfile()[index] - geo.getElevationProfile()[index + 1];
            return (drop / geo.getSpatialResolution()) * 100.0f;
        } else if (index > 0) {
            float drop = geo.getElevationProfile()[index - 1] - geo.getElevationProfile()[index];
            return (drop / geo.getSpatialResolution()) * 100.0f;
        }
        return 0.0f;
    }

    /**
     * Dibuja un panel informativo con estadísticas globales en la esquina superior izquierda.
     */
    private void drawMorphologyStatsOverlay(GraphicsContext gc, RiverGeometry geo) {
        // 1. CÁLCULO DE ESTADÍSTICAS
        float maxSlope = 0.0f;
        float maxZ = -Float.MAX_VALUE;
        float minZ = Float.MAX_VALUE;

        float[] elevs = geo.getElevationProfile();
        float[] slopes = geo.getSideSlope();
        int cells = geo.getCellCount();
        float dx = geo.getSpatialResolution();

        for (int i = 0; i < cells; i++) {
            // Taludes (z)
            float z = slopes[i];
            if (z > maxZ) maxZ = z;
            if (z < minZ) minZ = z;

            // Pendiente Longitudinal (%)
            if (i < cells - 1) {
                float drop = elevs[i] - elevs[i + 1];
                float currentSlopePct = (drop / dx) * 100.0f;
                if (currentSlopePct > maxSlope) maxSlope = currentSlopePct;
            }
        }

        float startElev = elevs[0];
        float endElev = elevs[cells - 1];

        // 2. DIBUJAR CAJA DE FONDO
        double x = 10;
        double y = 10;
        double width = 200;
        double height = 95; // Ajustado para 5 líneas

        gc.setFill(palette.bg().deriveColor(0, 1, 1, 0.8)); // Fondo semi-transparente
        gc.setStroke(palette.border());
        gc.setLineWidth(1.0);
        gc.fillRoundRect(x, y, width, height, 8, 8);
        gc.strokeRoundRect(x, y, width, height, 8, 8);

        // 3. DIBUJAR TEXTO
        gc.setFill(palette.text());
        gc.setFont(Font.font("Monospaced", FontWeight.BOLD, 11));

        double textX = x + 10;
        double textY = y + 20;
        double lineHeight = 16;

        gc.fillText(String.format("ELEV. INICIAL:  %7.1f m", startElev), textX, textY);
        textY += lineHeight;
        gc.fillText(String.format("ELEV. FINAL:    %7.1f m", endElev), textX, textY);
        textY += lineHeight;
        gc.fillText(String.format("PENDIENTE MAX:  %7.3f %%", maxSlope), textX, textY);
        textY += lineHeight + 4; // Separador visual

        // Datos de Sección
        gc.setFill(palette.text().deriveColor(0, 1, 1, 0.8)); // Un poco más apagado
        gc.fillText(String.format("TALUD (z) MAX:  %7.2f", maxZ), textX, textY);
        textY += lineHeight;
        gc.fillText(String.format("TALUD (z) MIN:  %7.2f", minZ), textX, textY);
    }

    private Color interpolateColor(Color c1, Color c2, double value, double min, double max) {
        double t = (value - min) / (max - min);
        t = Math.max(0, Math.min(1, t));
        return c1.interpolate(c2, t);
    }
}