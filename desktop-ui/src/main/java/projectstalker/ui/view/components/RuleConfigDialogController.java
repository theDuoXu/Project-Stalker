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
    @FXML
    private TableColumn<RuleViewModel, Double> colMin;
    @FXML
    private TableColumn<RuleViewModel, Double> colMax;

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
                // Commit on value change (Spinner arrows)
                spinner.valueProperty().addListener((obs, old, val) -> {
                    commit(val);
                });
                // Commit on focus lost (Text editing)
                spinner.getEditor().focusedProperty().addListener((obs, wasFocused, isNowFocused) -> {
                    if (!isNowFocused) {
                        commit(spinner.getValue());
                    }
                });
            }

            private void commit(Double val) {
                if (getIndex() >= 0 && getIndex() < getTableView().getItems().size()) {
                    getTableView().getItems().get(getIndex()).setThresholdSigma(val);
                }
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
                // Commit on value change
                spinner.valueProperty().addListener((obs, old, val) -> {
                    commit(val);
                });
                // Commit on focus lost
                spinner.getEditor().focusedProperty().addListener((obs, wasFocused, isNowFocused) -> {
                    if (!isNowFocused) {
                        commit(spinner.getValue());
                    }
                });
            }

            private void commit(Integer val) {
                if (getIndex() >= 0 && getIndex() < getTableView().getItems().size()) {
                    getTableView().getItems().get(getIndex()).setWindowSize(val);
                }
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

        configureMinMaxColumns();
    }

    private void configureDoubleColumn(TableColumn<RuleViewModel, Double> column,
            java.util.function.Function<RuleViewModel, javafx.beans.property.ObjectProperty<Double>> propertyExtractor) {

        column.setCellValueFactory(cellData -> propertyExtractor.apply(cellData.getValue()));
        column.setCellFactory(col -> new TableCell<>() {
            private final TextField textField = new TextField();

            {
                textField.focusedProperty().addListener((obs, wasFocused, isNowFocused) -> {
                    if (!isNowFocused) {
                        commitAndSave(textField.getText());
                    }
                });
                textField.setOnAction(e -> commitAndSave(textField.getText()));
            }

            private void commitAndSave(String text) {
                try {
                    Double val = Double.parseDouble(text);
                    if (getIndex() >= 0 && getIndex() < getTableView().getItems().size()) {
                        propertyExtractor.apply(getTableView().getItems().get(getIndex())).set(val);
                    }
                } catch (NumberFormatException e) {
                    // ignore invalid input, maybe reset to old val
                    if (getItem() != null) {
                        textField.setText(getItem().toString());
                    }
                }
            }

            @Override
            protected void updateItem(Double item, boolean empty) {
                super.updateItem(item, empty);
                if (empty) {
                    setGraphic(null);
                } else {
                    if (item != null) {
                        textField.setText(item.toString());
                    } else {
                        textField.setText("");
                    }
                    setGraphic(textField);
                }
            }
        });
    }

    private void configureMinMaxColumns() {
        configureDoubleColumn(colMin, RuleViewModel::minLimitProperty);
        configureDoubleColumn(colMax, RuleViewModel::maxLimitProperty);
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
                                .filter(d -> d.getMetric() != null)
                                .collect(java.util.stream.Collectors.toMap(RuleConfigDTO::getMetric, d -> d,
                                        (a, b) -> a));

                        for (String metric : allMetrics) {
                            if (existingMap.containsKey(metric)) {
                                rulesList.add(new RuleViewModel(existingMap.get(metric)));
                            } else {
                                // Defaults per user request
                                double min = 0.0;
                                double max = 100.0; // Default fallback
                                boolean isLog = "AMONIO".equals(metric) || "CONDUCTIVIDAD".equals(metric)
                                        || "TURBIDEZ".equals(metric);

                                switch (metric) {
                                    case "AMONIO" -> {
                                        min = 0.0;
                                        max = 20.0;
                                    }
                                    case "CARBONO ORGANICO" -> {
                                        min = 0.0;
                                        max = 50.0;
                                    }
                                    case "CLOROFILA" -> {
                                        min = 0.0;
                                        max = 500.0;
                                    }
                                    case "CONDUCTIVIDAD" -> {
                                        min = 20.0;
                                        max = 5000.0;
                                    }
                                    case "FICOCIANINAS" -> {
                                        min = 0.0;
                                        max = 1000.0;
                                    }
                                    case "FOSFATOS" -> {
                                        min = 0.0;
                                        max = 10.0;
                                    }
                                    case "NITRATOS" -> {
                                        min = 0.0;
                                        max = 250.0;
                                    }
                                    case "NIVEL" -> {
                                        min = 0.0;
                                        max = 20.0;
                                    }
                                    case "OXIGENO DISUELTO" -> {
                                        min = 0.0;
                                        max = 20.0;
                                    }
                                    case "PH" -> {
                                        min = 4.0;
                                        max = 10.5;
                                    }
                                    case "TEMPERATURA" -> {
                                        min = 0.0;
                                        max = 38.0;
                                    }
                                    case "TURBIDEZ" -> {
                                        min = 0.0;
                                        max = 2000.0;
                                    }
                                }

                                RuleViewModel vm = new RuleViewModel(
                                        new RuleConfigDTO(null, metric, isLog, 4.0, 5, min, max));
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

    private Runnable onSaveCallback;

    public void setOnSaveCallback(Runnable callback) {
        this.onSaveCallback = callback;
    }

    @FXML
    public void onSave() {
        // Collect DTOs first to avoid modification during iteration
        java.util.List<RuleConfigDTO> dtos = rulesList.stream()
                .map(RuleViewModel::toDTO)
                .filter(dto -> dto.getMetric() != null && !dto.getMetric().isEmpty())
                .toList();

        log.info("Saving {} rules...", dtos.size());

        // Use Flux to save all SEQUENTIALLY (concatMap) to avoid overloading the
        // backend
        reactor.core.publisher.Flux.fromIterable(dtos)
                .concatMap(dto -> {
                    log.info("Saving DTO: {}", dto.getMetric());
                    return ruleService.saveRule(dto);
                })
                .collectList() // Wait for all to complete
                .subscribe(savedList -> {
                    log.info("Successfully saved {} rules. Triggering refresh.", savedList.size());
                    javafx.application.Platform.runLater(() -> {
                        if (onSaveCallback != null) {
                            onSaveCallback.run();
                        }
                        onClose();
                    });
                }, err -> {
                    log.error("Error saving rules", err);
                    // Allow close anyway or show alert? For now log and close.
                    javafx.application.Platform.runLater(this::onClose);
                });
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
            this.id = dto.getId();
            this.metric.set(dto.getMetric());
            this.useLog.set(dto.isUseLog());
            this.thresholdSigma.set(dto.getThresholdSigma());
            this.windowSize.set(dto.getWindowSize());
            this.minLimit.set(dto.getMinLimit());
            this.maxLimit.set(dto.getMaxLimit());
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
