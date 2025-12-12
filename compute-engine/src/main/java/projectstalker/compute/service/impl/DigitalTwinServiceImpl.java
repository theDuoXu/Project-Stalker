package projectstalker.compute.service.impl;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import projectstalker.compute.entity.DigitalTwinEntity;
import projectstalker.compute.repository.DigitalTwinRepository;
import projectstalker.compute.service.DigitalTwinService;
import projectstalker.config.RiverConfig;
import projectstalker.domain.dto.twin.FlowPreviewRequest;
import projectstalker.domain.dto.twin.TwinCreateRequest;
import projectstalker.domain.dto.twin.TwinDetailDTO;
import projectstalker.domain.dto.twin.TwinSummaryDTO;
import projectstalker.domain.event.GeologicalEvent;
import projectstalker.domain.exception.ResourceNotFoundException;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.factory.RiverGeometryFactory;
import projectstalker.physics.model.RandomFlowProfileGenerator;
import projectstalker.physics.model.RiverTemperatureModel;

import java.time.LocalDateTime;
import java.util.List;
import java.util.UUID;

@Slf4j
@Service
@RequiredArgsConstructor
public class DigitalTwinServiceImpl implements DigitalTwinService {

    private final DigitalTwinRepository twinRepository;
    private final RiverGeometryFactory geometryFactory;

    // =========================================================================
    // 1. GESTIÓN DEL CICLO DE VIDA (PERSISTENCIA)
    // =========================================================================

    @Override
    @Transactional
    public TwinSummaryDTO createTwin(TwinCreateRequest request) {
        log.info("Creando nuevo Digital Twin: {}", request.name());

        // 1. Generar la geometría usando la Fábrica (CPU intensive)
        RiverGeometry geometry = generateFullGeometry(request.config(), request.events());

        // 2. Construir la Entidad
        DigitalTwinEntity entity = DigitalTwinEntity.builder()
                .id(UUID.randomUUID().toString())
                .name(request.name())
                .description(request.description())
                .createdAt(LocalDateTime.now())
                .config(request.config()) // JSONB
                .events(request.events()) // JSONB
                .geometry(geometry)       // JSONB (Pesado)
                .build();

        // 3. Guardar
        DigitalTwinEntity saved = twinRepository.save(entity);

        return mapToSummary(saved);
    }

    @Override
    @Transactional(readOnly = true)
    public List<TwinSummaryDTO> getAllTwins(int limit) {
        // Usamos PageRequest para limitar resultados y ordenar por fecha
        var pageable = PageRequest.of(0, limit, Sort.by(Sort.Direction.DESC, "createdAt"));

        return twinRepository.findAll(pageable)
                .stream()
                .map(this::mapToSummary)
                .toList();
    }

    @Override
    @Transactional(readOnly = true)
    public TwinDetailDTO getTwinDetails(String id) {
        DigitalTwinEntity entity = findEntityOrThrow(id);
        return mapToDetail(entity);
    }

    @Override
    @Transactional
    public TwinDetailDTO updateTwin(String id, TwinCreateRequest request) {
        log.info("Actualizando Digital Twin: {}", id);

        DigitalTwinEntity entity = findEntityOrThrow(id);

        // 1. Re-generar geometría (Vital si cambió la config o eventos)
        // Nota: Podríamos optimizar verificando si 'config' cambió realmente,
        // pero por seguridad regeneramos siempre para garantizar consistencia.
        RiverGeometry newGeometry = generateFullGeometry(request.config(), request.events());

        // 2. Actualizar campos
        entity.setName(request.name());
        entity.setDescription(request.description());
        entity.setConfig(request.config());
        entity.setEvents(request.events());
        entity.setGeometry(newGeometry);
        // entity.setUpdatedAt(LocalDateTime.now()); // Si tienes auditoría

        DigitalTwinEntity updated = twinRepository.save(entity);
        return mapToDetail(updated);
    }

    @Override
    @Transactional
    public void deleteTwin(String id) {
        if (!twinRepository.existsById(id)) {
            throw new ResourceNotFoundException("Digital Twin not found with id: " + id);
        }
        twinRepository.deleteById(id);
        log.info("Digital Twin eliminado: {}", id);
    }

    // =========================================================================
    // 2. MOTORES DE CÁLCULO (LIVE PREVIEW)
    // =========================================================================

    @Override
    public float[] previewTemperature(RiverConfig config, double timeOfDaySeconds) {
        // Para calcular temperatura necesitamos la geometría (ancho, pendiente...),
        // pero NO queremos guardar esto en BBDD.

        // 1. Generamos una geometría efímera en memoria (sin eventos geológicos para ser más rápido)
        // Ojo: Si la temperatura depende de presas, deberíamos pedir eventos también en el request.
        // Por ahora asumimos geometría base.
        RiverGeometry tempGeometry = geometryFactory.createRealisticRiver(config);

        // 2. Instanciamos el modelo físico
        RiverTemperatureModel model = new RiverTemperatureModel(config, tempGeometry);

        // 3. Calculamos
        return model.generateProfile(timeOfDaySeconds);
    }

    @Override
    public float[] previewFlow(FlowPreviewRequest request) {
        // 1. Instanciamos el generador de ruido
        RandomFlowProfileGenerator generator = new RandomFlowProfileGenerator(
                request.seed(),
                request.baseDischarge(),
                request.noiseAmplitude(),
                request.noiseFrequency()
        );

        // 2. Generamos la serie temporal
        // Usamos un timeStep arbitrario para la gráfica (ej: cada 60 segundos) o deducido.
        // Para previsualización, 300 puntos suele ser suficiente.
        double timeStep = request.durationSeconds() / 300.0;
        if (timeStep < 1.0) timeStep = 1.0;

        return generator.generateProfile(request.durationSeconds(), timeStep);
    }

    // =========================================================================
    // HELPERS PRIVADOS
    // =========================================================================

    private DigitalTwinEntity findEntityOrThrow(String id) {
        return twinRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("Digital Twin not found with id: " + id));
    }

    private RiverGeometry generateFullGeometry(RiverConfig config, List<GeologicalEvent> events) {
        // Paso 1: Río base procedural
        RiverGeometry base = geometryFactory.createRealisticRiver(config);

        // Paso 2: Aplicar eventos (presas, etc.)
        if (events != null && !events.isEmpty()) {
            return geometryFactory.applyGeologicalEvents(base, events);
        }
        return base;
    }

    // Mappers manuales (Evitamos MapStruct/ModelMapper para mantener control total)
    private TwinSummaryDTO mapToSummary(DigitalTwinEntity entity) {
        // Calculamos datos resumen desde la geometría
        float lengthKm = (entity.getGeometry().getCellCount() * entity.getGeometry().getSpatialResolution()) / 1000f;

        return TwinSummaryDTO.builder()
                .id(entity.getId())
                .name(entity.getName())
                .description(entity.getDescription())
                .createdAt(entity.getCreatedAt().toString())
                .totalLengthKm(lengthKm)
                .cellCount(entity.getGeometry().getCellCount())
                .build();
    }

    private TwinDetailDTO mapToDetail(DigitalTwinEntity entity) {
        return TwinDetailDTO.builder()
                .id(entity.getId())
                .name(entity.getName())
                .description(entity.getDescription())
                .config(entity.getConfig())
                .events(entity.getEvents())
                .geometry(entity.getGeometry())
                .build();
    }
}