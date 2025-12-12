package projectstalker.compute.service;

import projectstalker.config.RiverConfig;
import projectstalker.domain.dto.twin.FlowPreviewRequest;
import projectstalker.domain.dto.twin.TwinCreateRequest;
import projectstalker.domain.dto.twin.TwinDetailDTO;
import projectstalker.domain.dto.twin.TwinSummaryDTO;

import java.util.List;

/**
 * Servicio central para la gestión de Gemelos Digitales (Digital Twins).
 * <p>
 * Responsabilidades:
 * 1. Orquestar la creación y persistencia de ríos (CRUD).
 * 2. Invocar a la {@code RiverGeometryFactory} para generar geometrías complejas.
 * 3. Proveer capacidades de cálculo "al vuelo" (Live Preview) para herramientas de diseño.
 */
public interface DigitalTwinService {

    // =========================================================================
    // 1. GESTIÓN DEL CICLO DE VIDA (PERSISTENCIA)
    // =========================================================================

    /**
     * Crea un nuevo gemelo digital.
     * Genera la geometría completa usando la configuración proporcionada y lo guarda en base de datos.
     *
     * @param request Datos de creación (Configuración + Metadatos).
     * @return Un resumen del gemelo creado (sin la geometría pesada).
     */
    TwinSummaryDTO createTwin(TwinCreateRequest request);

    /**
     * Recupera el listado de gemelos digitales disponibles.
     *
     * @param limit Número máximo de elementos a recuperar (para optimizar carga).
     * @return Lista de DTOs ligeros.
     */
    List<TwinSummaryDTO> getAllTwins(int limit);

    /**
     * Recupera el detalle completo de un gemelo digital, incluyendo su geometría y configuración.
     *
     * @param id Identificador único del Twin.
     * @return DTO completo.
     * @throws projectstalker.domain.exception.ResourceNotFoundException si no existe.
     */
    TwinDetailDTO getTwinDetails(String id);

    /**
     * Actualiza la configuración de un gemelo digital existente.
     * <p>
     * IMPORTANTE: Esto provoca una <b>re-generación completa</b> de la geometría del río.
     *
     * @param id      Identificador del Twin a modificar.
     * @param request Nueva configuración y metadatos.
     * @return El DTO actualizado con la nueva geometría.
     */
    TwinDetailDTO updateTwin(String id, TwinCreateRequest request);

    /**
     * Elimina permanentemente un gemelo digital y sus simulaciones asociadas.
     *
     * @param id Identificador del Twin.
     */
    void deleteTwin(String id);

    // =========================================================================
    // 2. MOTORES DE CÁLCULO (LIVE PREVIEW - STATELESS)
    // =========================================================================

    /**
     * Calcula una vista previa del perfil de temperatura espacial.
     * No guarda nada en base de datos.
     *
     * @param config           Configuración del río (puede no estar guardada aún).
     * @param timeOfDaySeconds Hora del día en segundos para el cálculo solar.
     * @return Array con la temperatura en cada celda del río.
     */
    float[] previewTemperature(RiverConfig config, double timeOfDaySeconds);

    /**
     * Calcula una vista previa del hidrograma (perfil de caudal) generado por ruido.
     * No guarda nada en base de datos.
     *
     * @param request Parámetros del generador de flujo (caudal base, ruido, duración).
     * @return Array con el caudal (m3/s) en cada paso de tiempo.
     */
    float[] previewFlow(FlowPreviewRequest request);
}