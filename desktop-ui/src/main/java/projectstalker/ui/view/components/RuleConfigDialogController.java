package projectstalker.ui.view.components;

import javafx.beans.property.*;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;
import javafx.scene.control.*;
import javafx.scene.control.cell.CheckBoxTableCell;
import javafx.stage.Stage;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import projectstalker.domain.dto.rule.RuleConfigDTO;
import projectstalker.ui.service.RuleClientService;

@Slf4j
@Component
@RequiredArgsConstructor
public class RuleConfigDialogController {

    private final RuleClientService ruleService;

    @FXML
    private TableView<RuleViewModel> rulesTable;
    @FXML
    private TableColumn<RuleViewModel, String> colMetric;
    @FXML
    private TableColumn<RuleViewModel, Boolean> colLog;
    @FXML
    private TableColumn<RuleViewModel, Double> colThreshold;
    @FXML
    private TableColumn<RuleViewModel, Integer> colWindow;
    /*
     * @FXML
     * private TableColumn<RuleViewModel, Double> colMin;
     * 
     * @FXML
     * private TableColumn<RuleViewModel, Double> colMax;
     */

    private final ObservableList<RuleViewModel> rulesList = FXCollections.observableArrayList();

    @FXML
    public void initialize() {
        setupTable();
        loadRules();
    }

    private void setupTable() {
        rulesTable.setItems(rulesList);
        rulesTable.setEditable(true);

        colMetric.setCellValueFactory(d -> d.getValue().metricProperty());

        // Log Checkbox
        colLog.setCellValueFactory(d -> d.getValue().useLogProperty());
        colLog.setCellFactory(CheckBoxTableCell.forTableColumn(colLog));
        colLog.setEditable(true);

        // Threshold Spinner/Edit
        colThreshold.setCellValueFactory(d -> d.getValue().thresholdSigmaProperty().asObject());
        colThreshold.setCellFactory(col -> new TableCell<>() {
            private final Spinner<Double> spinner = new Spinner<>(2.0, 10.0, 4.0, 0.5);
            {
                spinner.setEditable(true);
                spinner.valueProperty().addListener((obs, old, val) -> {
                    if (getIndex() >= 0 && getIndex() < getTableView().getItems().size()) {
                        getTableView().getItems().get(getIndex()).setThresholdSigma(val);
                    }
                });
            }

            @Override
            protected void updateItem(Double item, boolean empty) {
                super.updateItem(item, empty);
                if (empty) {
                    setGraphic(null);
                } else {
                    spinner.getValueFactory().setValue(item);
                    setGraphic(spinner);
                }
            }
        });

        // Window Spinner
        colWindow.setCellValueFactory(d -> d.getValue().windowSizeProperty().asObject());
        colWindow.setCellFactory(col -> new TableCell<>() {
            private final Spinner<Integer> spinner = new Spinner<>(3, 200, 3, 1);
            {
                spinner.setEditable(true);
                spinner.valueProperty().addListener((obs, old, val) -> {
                    if (getIndex() >= 0 && getIndex() < getTableView().getItems().size()) {
                        getTableView().getItems().get(getIndex()).setWindowSize(val);
                    }
                });
            }

            @Override
            protected void updateItem(Integer item, boolean empty) {
                super.updateItem(item, empty);
                if (empty) {
                    setGraphic(null);
                } else {
                    spinner.getValueFactory().setValue(item);
                    setGraphic(spinner);
                }
            }
        });

    }

    private void loadRules() {
        ruleService.getAllRules()
                .collectList()
                .subscribe(list -> {
                    javafx.application.Platform.runLater(() -> {
                        rulesList.clear(); // Prevent duplicates on reload
                        java.util.List<String> allMetrics = java.util.List.of(
                                "PH", "AMONIO", "CLOROFILA", "CARBONO ORGANICO",
                                "OXIGENO DISUELTO", "CONDUCTIVIDAD", "TURBIDEZ", "TEMPERATURA",
                                "FICOCIANINAS", "NIVEL", "FOSFATOS", "NITRATOS");

                        java.util.Map<String, RuleConfigDTO> existingMap = list.stream()
                                .filter(d -> d.metric() != null)
                                .collect(java.util.stream.Collectors.toMap(RuleConfigDTO::metric, d -> d, (a, b) -> a)); // Fix
                                                                                                                         // Duplicate
                                                                                                                         // Key
                                                                                                                         // if
                                                                                                                         // DB
                                                                                                                         // has
                                                                                                                         // dupes

                        for (String metric : allMetrics) {
                            if (existingMap.containsKey(metric)) {
                                rulesList.add(new RuleViewModel(existingMap.get(metric)));
                            } else {
                                // Create default view model for missing metric
                                boolean isLog = "AMONIO".equals(metric) || "CONDUCTIVIDAD".equals(metric)
                                        || "TURBIDEZ".equals(metric); // Simplified guess
                                RuleViewModel vm = new RuleViewModel(
                                        new RuleConfigDTO(null, metric, isLog, 4.0, 5, null, null)); // Default 5 hours
                                rulesList.add(vm);
                            }
                        }
                    });
                }, err -> {
                    log.error("Error loading rules", err);
                    javafx.application.Platform.runLater(() -> {
                        rulesTable.setPlaceholder(new Label("Error de conexión con el servidor."));
                    });
                });
    }

    @FXML
    public void onSave() {
        // Save all
        rulesList.forEach(vm -> {
            RuleConfigDTO dto = vm.toDTO();
            ruleService.saveRule(dto)
                    .subscribe(saved -> log.info("Saved rule for {}", saved.metric()));
        });

        onClose();
    }

    @FXML
    public void onClose() {
        Stage stage = (Stage) rulesTable.getScene().getWindow();
        stage.close();
    }

    // Mutable VM
    public static class RuleViewModel {
        private final Long id;
        private final StringProperty metric = new SimpleStringProperty();
        private final BooleanProperty useLog = new SimpleBooleanProperty();
        private final DoubleProperty thresholdSigma = new SimpleDoubleProperty();
        private final IntegerProperty windowSize = new SimpleIntegerProperty();
        private final ObjectProperty<Double> minLimit = new SimpleObjectProperty<>();
        private final ObjectProperty<Double> maxLimit = new SimpleObjectProperty<>();

        public RuleViewModel(RuleConfigDTO dto) {
            this.id = dto.id();
            this.metric.set(dto.metric());
            this.useLog.set(dto.useLog());
            this.thresholdSigma.set(dto.thresholdSigma());
            this.windowSize.set(dto.windowSize());
            this.minLimit.set(dto.minLimit());
            this.maxLimit.set(dto.maxLimit());
        }

        public RuleConfigDTO toDTO() {
            return new RuleConfigDTO(id, metric.get(), useLog.get(), thresholdSigma.get(), windowSize.get(),
                    minLimit.get(), maxLimit.get());
        }

        // Getters for Properties
        public StringProperty metricProperty() {
            return metric;
        }

        public BooleanProperty useLogProperty() {
            return useLog;
        }

        public DoubleProperty thresholdSigmaProperty() {
            return thresholdSigma;
        }

        public IntegerProperty windowSizeProperty() {
            return windowSize;
        }

        public ObjectProperty<Double> minLimitProperty() {
            return minLimit;
        }

        public ObjectProperty<Double> maxLimitProperty() {
            return maxLimit;
        }

        public void setThresholdSigma(double v) {
            thresholdSigma.set(v);
        }

        public void setWindowSize(int v) {
            windowSize.set(v);
        }

        public void setMinLimit(Double v) {
            minLimit.set(v);
        }

        public void setMaxLimit(Double v) {
            maxLimit.set(v);
        }
    }
}
