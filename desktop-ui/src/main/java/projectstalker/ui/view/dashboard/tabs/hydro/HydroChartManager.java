package projectstalker.ui.view.dashboard.tabs.hydro;

import javafx.application.Platform;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.XYChart;
import javafx.scene.chart.NumberAxis;
import projectstalker.domain.dto.sensor.SensorReadingDTO;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

@lombok.extern.slf4j.Slf4j
public class HydroChartManager {

    private final LineChart<String, Number> chart;
    private final XYChart.Series<String, Number> series;

    public HydroChartManager(LineChart<String, Number> chart) {
        this.chart = chart;
        this.series = new XYChart.Series<>();
        setupChart();
    }

    private void setupChart() {
        chart.setAnimated(false);
        chart.getData().add(series);
        series.setName("Nivel");
    }

    public void setMetricName(String metric) {
        series.setName(metric);
        if (chart.getYAxis() instanceof NumberAxis) {
            chart.getYAxis().setLabel("Valor (" + metric + ")");
        }
    }

    public String getCurrentMetric() {
        return series.getName();
    }

    public void clear() {
        series.getData().clear();
    }

    public void addPoint(String label, Number value) {
        Platform.runLater(() -> {
            series.getData().add(new XYChart.Data<>(label, value));
            // Keep limit
            if (series.getData().size() > 50) {
                series.getData().remove(0);
            }
        });
    }

    public void updateData(List<SensorReadingDTO> readings, String metric, boolean forcePlot) {
        Platform.runLater(() -> {
            List<SensorReadingDTO> filtered = readings.stream()
                    .filter(r -> forcePlot || metric.equals(r.tag())
                            || ("value".equals(metric) && "value".equals(r.tag())))
                    .sorted(Comparator.comparing(SensorReadingDTO::timestamp))
                    .collect(Collectors.toList());

            log.info("HydroChartManager: Received {} readings. Filtered: {}", readings.size(), filtered.size());

            setMetricName(metric);
            series.getData().clear();

            for (SensorReadingDTO r : filtered) {
                String label = formatTimeLabel(r.timestamp());
                series.getData().add(new XYChart.Data<>(label, r.value()));
            }
        });
    }

    private String formatTimeLabel(String isoTimestamp) {
        try {
            LocalDateTime dt = LocalDateTime.parse(isoTimestamp);
            return dt.format(DateTimeFormatter.ofPattern("dd/MM HH:mm"));
        } catch (Exception e) {
            // Fallback for short times or errors
            if (isoTimestamp.length() >= 16)
                return isoTimestamp.substring(11, 16);
            return isoTimestamp;
        }
    }
}
