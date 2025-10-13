package projectstalker.domain.event;

import projectstalker.domain.river.RiverSectionType;

/**
 * Representa un evento que modifica los perfiles de una geometría de río.
 * Cada implementación define su propio impacto sobre los arrays de datos.
 */
public interface GeologicalEvent {

    /**
     * Aplica el efecto del evento directamente sobre los arrays de datos de un río.
     * <p>
     * <b>Nota para implementadores:</b> Este método está diseñado para operar
     * sobre copias mutables de los datos originales del río.
     *
     * @param spatialResolution  La resolución espacial (distancia entre celdas) en metros.
     * @param elevationProfile   El array del perfil de elevaciones a modificar.
     * @param bottomWidth        El array del perfil de anchuras a modificar.
     * @param manningCoefficient El array del coeficiente de Manning a modificar.
     */
    void apply(
            double spatialResolution,
            double[] elevationProfile,
            double[] bottomWidth,
            double[] manningCoefficient,
            RiverSectionType[] sectionTypes
    );
}