package projectstalker.physics.model;

import lombok.extern.slf4j.Slf4j;
import projectstalker.domain.river.RiverGeometry;

/**
 * Modela el perfil de pH a lo largo del río.
 * En la versión actual, el pH es estático o se basa en la geometría inicial
 * y no varía con el tiempo.
 */
@Slf4j
public class RiverPhModel {

    private final RiverGeometry geometry;

    /**
     * Constructor.
     * @param geometry La geometría del río de donde se obtiene el perfil de pH inicial.
     */
    public RiverPhModel(RiverGeometry geometry) {
        this.geometry = geometry;
        log.info("RiverPhModel inicializado. Usando el perfil de pH base de RiverGeometry.");
    }

    /**
     * Devuelve el perfil actual de pH del río.
     * En la simulación hidrológica de Manning, se asume que el pH es un perfil
     * estático (o pre-calculado) que viaja con la onda de agua, pero no se calcula
     * su dinámica internamente. Por ello, solo clona el perfil base.
     *
     * @return Un nuevo array que representa el perfil de pH del río.
     */
    public float[] getPhProfile() {
        return geometry.clonePhProfile();
    }
}