package projectstalker.compute.api;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;
import projectstalker.compute.service.InferenceService;

import java.util.Map;

@RestController
@RequestMapping("/inference")
@RequiredArgsConstructor
@Tag(name = "Inferencia (DeepONet)", description = "Predicciones del modelo DeepONet")
public class InferenceController {

    private final InferenceService inferenceService;

    @PostMapping("/mock/config")
    @Operation(summary = "Configurar Sensor Mock", description = "Guarda la configuración del río y devuelve un hash.")
    public String configureMock(@RequestBody Map<String, Object> riverConfig) {
        return inferenceService.configureMock(riverConfig);
    }

    @PostMapping("/run")
    @Operation(summary = "Ejecutar Inferencia", description = "Llama al motor de inferencia (o mock) y devuelve el mapa de calor.")
    public Map<String, Object> runInference(@RequestBody Map<String, Object> request) {
        return inferenceService.runInference(request);
    }
}
