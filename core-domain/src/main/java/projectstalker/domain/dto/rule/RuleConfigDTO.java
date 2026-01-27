package projectstalker.domain.dto.rule;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@com.fasterxml.jackson.annotation.JsonView(projectstalker.domain.sensors.SensorViews.Public.class)
public class RuleConfigDTO {
        @JsonProperty("id")
        private Long id;

        @JsonProperty("metric")
        private String metric;

        @JsonProperty("useLog")
        private boolean useLog;

        @JsonProperty("thresholdSigma")
        private double thresholdSigma;

        @JsonProperty("windowSize")
        private int windowSize;

        @JsonProperty("minLimit")
        private Double minLimit;

        @JsonProperty("maxLimit")
        private Double maxLimit;
}
