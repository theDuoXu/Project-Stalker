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
import projectstalker.domain.river.HydrologySnapshot;
import projectstalker.domain.river.RiverGeometry;

import java.util.Arrays;

/**
 * Renderizador del río con soporte para HUD flotante, Modos de Visualización y Simulación Hidrológica.
 */
public class RiverRenderer {

    private final Canvas canvas;
    private ThemePalette palette;


    private record ThemePalette(
            Color bg, Color text, Color border,
            Color waterDeep, Color waterSurface,
            Color sand, Color rock,
            Color acidZone, Color baseZone,
            Color cursor,
            Color analysisLine1, Color analysisLine2,
            Color hydroTemp, Color hydroPh, Color hydroDecay, Color gridLines
    ) {}

    @Getter
    public enum RenderMode {
        MORPHOLOGY(false, "Vista realista (Morfología)"),
        ANALYSIS(true, "Vista de datos (Estática)"),
        HYDROLOGY(false, "Simulación en Tiempo Real"); // No es analítico estático, es dinámico

        private final boolean isAnalyticalMode;
        private final String title;

        RenderMode(boolean isAnalyticalMode, String title) {
            this.isAnalyticalMode = isAnalyticalMode;
            this.title = title;
        }

        // Helper para el ToggleButton de la pestaña 1 (Morfología vs Análisis)
        public static RenderMode fromBoolean(boolean isAnalyticalMode) {
            if (isAnalyticalMode) return ANALYSIS;
            return MORPHOLOGY;
        }
    }

    public RiverRenderer(Canvas canvas) {
        this.canvas = canvas;
        this.palette = createFallbackPalette();
    }

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

                resolveColor(root, "-color-danger-fg", Color.web("#FF5555")),
                resolveColor(root, "-color-success-fg", Color.web("#50FA7B")),

                resolveColor(root, "-color-fg-default", Color.WHITE).deriveColor(0, 1, 1, 0.8),

                Color.web("#F8F8F2"), // Ancho
                Color.web("#BD93F9"), // Talud

                // Colores Hidrología (Neon Style)
                Color.web("#FF5555"), // Temp (Rojo)
                Color.web("#50FA7B"), // pH (Verde)
                Color.web("#8BE9FD"), // Decay (Cyan)
                resolveColor(root, "-color-border-subtle", Color.web("#4C566A")) // Grid
        );
    }

    private ThemePalette createFallbackPalette() {
        return new ThemePalette(
                Color.web("#2E3440"), Color.WHITE, Color.GRAY,
                Color.BLUE, Color.LIGHTBLUE, Color.SANDYBROWN, Color.DARKGRAY,
                Color.RED, Color.GREEN, Color.WHITE,
                Color.WHITE, Color.MAGENTA,
                Color.RED, Color.GREEN, Color.CYAN, Color.DARKGRAY
        );
    }

    private Color resolveColor(Node root, String cssVariable, Color fallback) {
        try {
            Label dummy = new javafx.scene.control.Label();
            dummy.setVisible(false); dummy.setManaged(false);
            javafx.scene.layout.Pane container = null;
            if (root instanceof javafx.scene.layout.Pane) container = (javafx.scene.layout.Pane) root;
            else if (canvas.getParent() instanceof javafx.scene.layout.Pane) container = (javafx.scene.layout.Pane) canvas.getParent();
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
     * Render principal (Morfología y Análisis).
     */
    public void render(RiverGeometry geo, RenderMode mode, double mouseX, double mouseY, Runnable onRenderFinished) {
        // Si intentan usar este método para Hidrología sin snapshot, no hacemos nada o fallback
        if (mode == RenderMode.HYDROLOGY) return;

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

            if (onRenderFinished != null) onRenderFinished.run();
        });
    }

    /**
     * Render específico para Simulación Hidrológica (Requiere Snapshot).
     */
    public void renderHydrology(Canvas targetCanvas, RiverGeometry geo, HydrologySnapshot snap, double mouseX, double mouseY) {
        Platform.runLater(() -> {
            GraphicsContext gc = targetCanvas.getGraphicsContext2D();
            double w = targetCanvas.getWidth();
            double h = targetCanvas.getHeight();

            gc.clearRect(0, 0, w, h);
            gc.setFill(palette.bg());
            gc.fillRect(0, 0, w, h);

            if (geo != null && snap != null) {
                drawHydrologyView(gc, geo, snap, w, h);

                if (mouseX >= 0 && mouseX < w && mouseY >= 0 && mouseY < h) {
                    drawHydrologyHUD(gc, geo, snap, mouseX, mouseY, w, h);
                }
            }
        });
    }

    // Sobrecarga simple
    public void render(RiverGeometry geo, RenderMode mode, double mouseX, double mouseY) {
        render(geo, mode, mouseX, mouseY, null);
    }

    // -------------------------------------------------------------------------
    // VISTA 1: MORFOLOGÍA (Planta)
    // -------------------------------------------------------------------------
    private void drawPlanView(GraphicsContext gc, RiverGeometry geo, double w, double h) {
        float[] widths = geo.getBottomWidth();
        int cells = geo.getCellCount();
        double scaleX = w / cells;
        double centerY = h / 2.0;

        double maxRiverWidth = 0;
        for (float width : widths) maxRiverWidth = Math.max(maxRiverWidth, width);
        double viewportMeters = Math.max(maxRiverWidth, 250.0);
        double scaleY = (h * 0.9) / viewportMeters;

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

    // -------------------------------------------------------------------------
    // VISTA 2: ANÁLISIS (Gráficas Estáticas)
    // -------------------------------------------------------------------------
    private void drawAnalysisView(GraphicsContext gc, RiverGeometry geo, double w, double h) {
        int cells = geo.getCellCount();
        float[] mannings = geo.getManningCoefficient();
        float[] phs = geo.getPhProfile();
        float[] decays = geo.getBaseDecayCoefficientAt20C();
        float[] widths = geo.getBottomWidth();
        float[] slopes = geo.getSideSlope();

        double scaleX = w / cells;
        double centerY = h / 2.0;

        // Contexto Fantasma
        double maxRiverWidth = 0;
        for (float width : widths) maxRiverWidth = Math.max(maxRiverWidth, width);
        double viewportMeters = Math.max(maxRiverWidth, 250.0);
        double scaleYRiver = (h * 0.9) / viewportMeters;

        gc.setFill(palette.waterSurface().deriveColor(0, 1, 1, 0.1));
        for (int i = 0; i < cells; i++) {
            double x = i * scaleX;
            double riverW = widths[i] * scaleYRiver;
            gc.fillRect(x, centerY - (riverW / 2), Math.max(1.0, scaleX), riverW);
        }

        // Gráficas
        double margin = 30;
        double gH = h - (margin * 2);

        double[] xP = new double[cells];
        double[] yWidth = new double[cells];
        double[] ySlope = new double[cells];
        double[] yManning = new double[cells];
        double[] yPh = new double[cells];
        double[] yDecay = new double[cells];

        for (int i = 0; i < cells; i++) {
            xP[i] = i * scaleX;
            yWidth[i] = (h - margin) - ((widths[i] / (maxRiverWidth * 1.1)) * gH);
            ySlope[i] = (h - margin) - ((slopes[i] / 5.0) * gH);

            double normM = (mannings[i] - 0.02) / 0.04;
            yManning[i] = (h - margin) - (Math.max(0, Math.min(1, normM)) * gH);

            double normPh = (phs[i] - 6.0) / 3.0;
            yPh[i] = (h - margin) - (Math.max(0, Math.min(1, normPh)) * gH);

            yDecay[i] = (h - margin) - ((decays[i] / 0.5) * gH);
        }

        gc.setLineWidth(2.0);
        drawPolyline(gc, xP, yWidth, cells, palette.analysisLine1());
        drawPolyline(gc, xP, ySlope, cells, palette.analysisLine2());
        drawPolyline(gc, xP, yManning, cells, palette.sand());
        drawPolyline(gc, xP, yPh, cells, palette.acidZone());
        drawPolyline(gc, xP, yDecay, cells, palette.waterDeep().brighter());

        drawAnalysisLegend(gc, 10, 20, palette.analysisLine1(), palette.analysisLine2(), palette.sand(), palette.acidZone(), palette.waterDeep().brighter());
    }

    // -------------------------------------------------------------------------
    // VISTA 3: HIDROLOGÍA (Gráficas Dinámicas X-Y)
    // -------------------------------------------------------------------------
    private void drawHydrologyView(GraphicsContext gc, RiverGeometry geo, HydrologySnapshot snap, double w, double h) {
        int cells = geo.getCellCount();
        double scaleX = w / cells;
        double margin = 40; // Margen para ejes
        double graphH = h - (margin * 2);

        // 1. DIBUJAR GRID Y EJES
        drawHydrologyGrid(gc, w, h, margin, graphH, geo.getTotalLength());

        // 2. NORMALIZAR Y DIBUJAR CURVAS
        double[] xPoints = new double[cells];
        double[] yTemp = new double[cells];
        double[] yPh = new double[cells];
        double[] yDecay = new double[cells];

        for (int i = 0; i < cells; i++) {
            xPoints[i] = margin + (i * scaleX * ((w - margin * 2) / w)); // Ajuste por margen

            // Temp: 0°C - 35°C
            double normTemp = snap.temperature()[i] / 35.0;
            yTemp[i] = (h - margin) - (Math.max(0, Math.min(1, normTemp)) * graphH);

            // pH: 0 - 14
            double normPh = snap.ph()[i] / 14.0;
            yPh[i] = (h - margin) - (Math.max(0, Math.min(1, normPh)) * graphH);

            // Decay: 0 - 1.0 y multiplicamos por 2 visualmente porque suele ser bajo
            double normDecay = snap.decay()[i];
            yDecay[i] = (h - margin) - (Math.max(0, Math.min(1, normDecay)) * graphH);
        }

        gc.setLineWidth(2.0);
        // Temp
        gc.setStroke(palette.hydroTemp());
        gc.strokePolyline(xPoints, yTemp, cells);

        // pH
        gc.setStroke(palette.hydroPh());
        gc.strokePolyline(xPoints, yPh, cells);

        // Decay
        gc.setStroke(palette.hydroDecay());
        gc.strokePolyline(xPoints, yDecay, cells);

        // Leyenda
        drawHydrologyLegend(gc, w - 160, 20);
    }

    private void drawHydrologyGrid(GraphicsContext gc, double w, double h, double m, double gH, float totalLengthM) {
        gc.setStroke(palette.gridLines());
        gc.setLineWidth(0.5);
        gc.setLineDashes(5);
        gc.setFill(palette.text().deriveColor(0,1,1,0.5));
        gc.setFont(Font.font("Arial", 10));

        // Líneas Horizontales (0%, 25%, 50%, 75%, 100%)
        for (int i = 0; i <= 4; i++) {
            double y = (h - m) - (i * 0.25 * gH);
            gc.strokeLine(m, y, w - m, y);
            // Etiquetas Ejes (Temp | pH)
            gc.fillText(String.format("%.0f°C", i * 8.75), 5, y + 4);
            gc.fillText(String.format("%.1f", i * 3.5), w - m + 5, y + 4);
        }

        // Eje X (Distancia)
        gc.fillText("0 km", m, h - 5);
        gc.fillText(String.format("%.1f km", totalLengthM / 1000.0), w - m - 30, h - 5);

        gc.setLineDashes(0);

        // Marco
        gc.setStroke(palette.border());
        gc.strokeRect(m, m, w - (m*2), gH);
    }

    // -------------------------------------------------------------------------
    // HUDs
    // -------------------------------------------------------------------------
    private void drawFloatingHUD(GraphicsContext gc, RiverGeometry geo, RenderMode mode, double mx, double my, double w, double h) {
        int cellCount = geo.getCellCount();
        double pixelsPerCell = w / cellCount;
        int index = (int) (mx / pixelsPerCell);

        if (index < 0) index = 0;
        if (index >= cellCount) index = cellCount - 1;

        // Cursor
        double cursorX = index * pixelsPerCell + (pixelsPerCell / 2.0);
        gc.setStroke(palette.cursor());
        gc.setLineWidth(1.0);
        gc.setLineDashes(4);
        gc.strokeLine(cursorX, 0, cursorX, h);
        gc.setLineDashes(0);

        // Caja
        double boxW = 220;
        double boxH = (mode == RenderMode.MORPHOLOGY) ? 160 : 190;
        double padding = 15;
        double boxX = (mx + padding + boxW > w) ? (mx - boxW - padding) : (mx + padding);
        double boxY = (my + padding + boxH > h) ? (my - boxH - padding) : (my + padding);

        drawHUDBox(gc, boxX, boxY, boxW, boxH);

        // Header
        float dist = (float) (index * geo.getSpatialResolution());
        float elev = geo.getElevationProfile()[index];
        drawHUDHeader(gc, boxX, boxY, dist, elev);

        double contentY = boxY + 25 + 18 + 18 + 5;
        double textX = boxX + 15;

        if (mode == RenderMode.MORPHOLOGY) {
            drawMorphologyHUDContent(gc, geo, index, boxX, boxW, boxY, boxH, contentY, textX);
        } else {
            drawAnalysisHUDContent(gc, geo, index, textX, contentY);
        }
    }

    private void drawHydrologyHUD(GraphicsContext gc, RiverGeometry geo, HydrologySnapshot snap, double mx, double my, double w, double h) {
        double margin = 40;
        // Ajustar mouseX al área del gráfico
        if (mx < margin || mx > w - margin) return;

        double graphW = w - (margin * 2);
        int cellCount = geo.getCellCount();
        double pixelsPerCell = graphW / cellCount;
        int index = (int) ((mx - margin) / pixelsPerCell);

        if (index < 0) index = 0; if (index >= cellCount) index = cellCount - 1;

        // Cursor
        double cursorX = margin + (index * pixelsPerCell);
        gc.setStroke(palette.cursor());
        gc.setLineWidth(1.0);
        gc.setLineDashes(4);
        gc.strokeLine(cursorX, margin, cursorX, h - margin);
        gc.setLineDashes(0);

        // Caja
        double boxW = 220; double boxH = 130;
        double boxX = (mx + 15 + boxW > w) ? (mx - boxW - 15) : (mx + 15);
        double boxY = my + 15; if (boxY + boxH > h) boxY = h - boxH - 10;

        drawHUDBox(gc, boxX, boxY, boxW, boxH);

        float dist = (float) (index * geo.getSpatialResolution());
        // Header simple
        gc.setFill(palette.text());
        gc.setFont(Font.font("Monospaced", FontWeight.BOLD, 12));
        gc.fillText(String.format("DIST: %.0f m", dist), boxX + 15, boxY + 25);
        gc.setStroke(Color.GRAY);
        gc.strokeLine(boxX+15, boxY+35, boxX+boxW-15, boxY+35);

        // Datos Dinámicos
        double startY = boxY + 55;
        double ls = 18;

        drawDataRow(gc, boxX+15, startY, palette.hydroTemp(), "Temp:", String.format("%.2f °C", snap.temperature()[index])); startY += ls;
        drawDataRow(gc, boxX+15, startY, palette.hydroPh(),   "pH:",   String.format("%.2f", snap.ph()[index])); startY += ls;
        drawDataRow(gc, boxX+15, startY, palette.hydroDecay(),"Decay:",String.format("%.3f", snap.decay()[index]));
    }

    // -------------------------------------------------------------------------
    // HELPERS DIBUJO
    // -------------------------------------------------------------------------
    private void drawHUDBox(GraphicsContext gc, double x, double y, double w, double h) {
        gc.setFill(palette.bg().deriveColor(0, 1, 1, 0.95));
        gc.setStroke(palette.border());
        gc.setLineWidth(1.5);
        gc.fillRoundRect(x, y, w, h, 10, 10);
        gc.strokeRoundRect(x, y, w, h, 10, 10);
    }

    private void drawHUDHeader(GraphicsContext gc, double boxX, double boxY, float dist, float elev) {
        gc.setFill(palette.text());
        gc.setFont(Font.font("Monospaced", FontWeight.BOLD, 12));
        double textX = boxX + 15;
        double lineH = 18;
        double curY = boxY + 25;
        gc.fillText(String.format("DIST: %.0f m", dist), textX, curY); curY += lineH;
        gc.fillText(String.format("ELEV: %.2f m", elev), textX, curY); curY += lineH + 5;
        gc.setStroke(Color.GRAY);
        gc.setLineWidth(0.5);
        gc.strokeLine(textX, curY - 5, boxX + 220 - 15, curY - 5);
    }

    private void drawMorphologyHUDContent(GraphicsContext gc, RiverGeometry geo, int index,
                                          double boxX, double boxW, double boxY, double boxH,
                                          double startY, double textX) {
        float bottomWidth = geo.getBottomWidth()[index];
        float z = geo.getSideSlope()[index];
        float slopePercent = calculateLocalSlopePercent(geo, index);

        gc.setFill(palette.text());
        gc.fillText(String.format("PEND: %.3f %%", slopePercent), textX, startY);

        // Mini Cross-Section
        double graphH = 50; double graphW = boxW - 30;
        double graphBaseY = boxY + boxH - 15;
        double graphCenterX = boxX + (boxW / 2.0);
        double visualDepth = graphH * 0.8;
        double visualBaseWidth = Math.min(graphW * 0.4, bottomWidth * 2.0);
        double visualTopWidth = visualBaseWidth + (2 * z * (visualDepth / 2.0));

        double xBL = graphCenterX - (visualBaseWidth/2.0); double xBR = graphCenterX + (visualBaseWidth/2.0);
        double xTL = graphCenterX - (visualTopWidth/2.0); double xTR = graphCenterX + (visualTopWidth/2.0);
        double yB = graphBaseY; double yT = graphBaseY - visualDepth;

        gc.beginPath(); gc.moveTo(xBL, yB); gc.lineTo(xTL, yT); gc.lineTo(xTR, yT); gc.lineTo(xBR, yB); gc.closePath();
        LinearGradient grad = new LinearGradient(0, 0, 0, 1, true, CycleMethod.NO_CYCLE,
                new Stop(0, palette.waterSurface()), new Stop(1, palette.waterDeep));
        gc.setFill(grad); gc.fill();

        gc.setStroke(palette.text().deriveColor(0, 1, 1, 0.5));
        gc.setLineWidth(2.0);
        gc.strokePolyline(new double[]{xTL, xBL, xBR, xTR}, new double[]{yT, yB, yB, yT}, 4);

        gc.setFont(Font.font("Arial", 9));
        gc.setFill(palette.text().deriveColor(0, 1, 1, 0.7));
        String shape = (z < 1.0) ? "V (Cañón)" : "U (Valle)";
        gc.fillText(shape, xTR + 5, yT + 10);
    }

    private void drawAnalysisHUDContent(GraphicsContext gc, RiverGeometry geo, int index, double x, double startY) {
        float w = geo.getBottomWidth()[index];
        float z = geo.getSideSlope()[index];
        float n = geo.getManningCoefficient()[index];
        float ph = geo.getPhProfile()[index];
        float k = geo.getBaseDecayCoefficientAt20C()[index];
        double y = startY; double ls = 18;
        drawDataRow(gc, x, y, palette.analysisLine1(), "Ancho (b):", String.format("%.2f m", w)); y += ls;
        drawDataRow(gc, x, y, palette.analysisLine2(), "Talud (z):", String.format("%.2f", z));   y += ls;
        drawDataRow(gc, x, y, palette.sand(),          "Manning (n):", String.format("%.4f", n)); y += ls;
        drawDataRow(gc, x, y, palette.acidZone(),      "pH:",          String.format("%.2f", ph));y += ls;
        drawDataRow(gc, x, y, palette.waterDeep().brighter(), "Decay (k):", String.format("%.3f", k));
    }

    private void drawDataRow(GraphicsContext gc, double x, double y, Color c, String label, String value) {
        gc.setFill(c); gc.fillOval(x, y - 8, 8, 8);
        gc.setFill(palette.text()); gc.setFont(Font.font("Monospaced", FontWeight.NORMAL, 12));
        gc.fillText(label, x + 15, y);
        gc.setFont(Font.font("Monospaced", FontWeight.BOLD, 12));
        gc.fillText(value, x + 120, y);
    }

    private void drawAnalysisLegend(GraphicsContext gc, double x, double y, Color... colors) {
        gc.setFill(palette.bg().deriveColor(0, 1, 1, 0.9));
        gc.setStroke(palette.border());
        gc.fillRoundRect(x, y, 140, 110, 10, 10);
        gc.strokeRoundRect(x, y, 140, 110, 10, 10);
        gc.setFont(Font.font("Arial", FontWeight.BOLD, 10));
        double ly = y + 20;
        drawLegendItem(gc, x + 10, ly, colors[0], "Ancho (b)"); ly += 18;
        drawLegendItem(gc, x + 10, ly, colors[1], "Talud (z)"); ly += 18;
        drawLegendItem(gc, x + 10, ly, colors[2], "Manning (n)"); ly += 18;
        drawLegendItem(gc, x + 10, ly, colors[3], "pH (6-9)"); ly += 18;
        drawLegendItem(gc, x + 10, ly, colors[4], "Decay (k)");
    }

    private void drawHydrologyLegend(GraphicsContext gc, double x, double y) {
        gc.setFill(palette.bg().deriveColor(0, 1, 1, 0.9));
        gc.setStroke(palette.border());
        gc.fillRoundRect(x, y, 140, 80, 10, 10);
        gc.strokeRoundRect(x, y, 140, 80, 10, 10);
        gc.setFont(Font.font("Arial", FontWeight.BOLD, 10));
        double ly = y + 20;
        drawLegendItem(gc, x + 10, ly, palette.hydroTemp(), "Temp (0-35°C)"); ly += 18;
        drawLegendItem(gc, x + 10, ly, palette.hydroPh(), "pH (0-14)"); ly += 18;
        drawLegendItem(gc, x + 10, ly, palette.hydroDecay(), "Decay");
    }

    private void drawLegendItem(GraphicsContext gc, double x, double y, Color c, String text) {
        gc.setStroke(c); gc.setLineWidth(2.0);
        gc.strokeLine(x, y, x + 15, y);
        gc.setFill(palette.text()); gc.fillText(text, x + 25, y + 4);
    }

    private void drawPolyline(GraphicsContext gc, double[] x, double[] y, int pts, Color c) {
        gc.setStroke(c); gc.strokePolyline(x, y, pts);
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

    private void drawMorphologyStatsOverlay(GraphicsContext gc, RiverGeometry geo) {
        float maxSlope = 0.0f; float maxZ = -Float.MAX_VALUE; float minZ = Float.MAX_VALUE;
        float maxWidth = -Float.MAX_VALUE; float minWidth = Float.MAX_VALUE;
        float[] elevs = geo.getElevationProfile(); float[] slopes = geo.getSideSlope();
        float[] widths = geo.getBottomWidth(); int cells = geo.getCellCount(); float dx = geo.getSpatialResolution();

        for (int i = 0; i < cells; i++) {
            float z = slopes[i]; if (z > maxZ) maxZ = z; if (z < minZ) minZ = z;
            float w = widths[i]; if (w > maxWidth) maxWidth = w; if (w < minWidth) minWidth = w;
            if (i < cells - 1) {
                float drop = elevs[i] - elevs[i + 1];
                float currentSlopePct = (drop / dx) * 100.0f;
                if (currentSlopePct > maxSlope) maxSlope = currentSlopePct;
            }
        }
        float startElev = elevs[0]; float endElev = elevs[cells - 1];
        double x = 10; double y = 10; double boxW = 210; double boxH = 135;
        drawHUDBox(gc, x, y, boxW, boxH);
        gc.setFill(palette.text()); gc.setFont(Font.font("Monospaced", FontWeight.BOLD, 11));
        double textX = x + 10; double textY = y + 20; double lineHeight = 16;
        gc.fillText(String.format("ELEV. INICIAL:  %7.1f m", startElev), textX, textY); textY += lineHeight;
        gc.fillText(String.format("ELEV. FINAL:    %7.1f m", endElev), textX, textY); textY += lineHeight;
        gc.fillText(String.format("PENDIENTE MAX:  %7.3f %%", maxSlope), textX, textY); textY += lineHeight + 5;
        gc.fillText(String.format("ANCHO MAX:      %7.1f m", maxWidth), textX, textY); textY += lineHeight;
        gc.fillText(String.format("ANCHO MIN:      %7.1f m", minWidth), textX, textY); textY += lineHeight + 5;
        gc.setFill(palette.text().deriveColor(0, 1, 1, 0.8));
        gc.fillText(String.format("TALUD (z) MAX:  %7.2f", maxZ), textX, textY); textY += lineHeight;
        gc.fillText(String.format("TALUD (z) MIN:  %7.2f", minZ), textX, textY);
    }
}