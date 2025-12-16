package projectstalker.ui.view.util;

import javafx.scene.control.Spinner;
import javafx.scene.control.SpinnerValueFactory;
import javafx.util.StringConverter;

public class RiverUiFactory {

    public static void configureSpinner(Spinner<Double> spinner, double min, double max, double initial, double step) {
        spinner.setValueFactory(new SpinnerValueFactory.DoubleSpinnerValueFactory(min, max, initial, step));
    }

    public static void configureSpinner(Spinner<Integer> spinner, int min, int max, int initial, int step) {
        spinner.setValueFactory(new SpinnerValueFactory.IntegerSpinnerValueFactory(min, max, initial, step));
    }

    // Tu lógica compleja de formateo encapsulada aquí
    public static void configureScientificSpinner(Spinner<Double> spinner, double min, double max, double initial, double step, String format) {
        var factory = new SpinnerValueFactory.DoubleSpinnerValueFactory(min, max, initial, step);
        factory.setConverter(createConverter(format, initial));
        spinner.setValueFactory(factory);
    }

    private static StringConverter<Double> createConverter(String format, double fallbackValue) {
        return new StringConverter<>() {
            @Override
            public String toString(Double object) {
                if (object == null) return String.format(format, fallbackValue);
                return String.format(format, object);
            }

            @Override
            public Double fromString(String string) {
                try {
                    return Double.parseDouble(string.replace(",", "."));
                } catch (NumberFormatException e) {
                    return fallbackValue;
                }
            }
        };
    }
}