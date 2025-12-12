package projectstalker.domain.event;

import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import projectstalker.domain.river.RiverSectionType;

/**
 * Representa un evento que modifica los perfiles de una geometría de río.
 * Cada implementación define su propio impacto sobre los arrays de datos.
 */
// 1. Decimos que vamos a usar el NOMBRE como discriminador
@JsonTypeInfo(
        use = JsonTypeInfo.Id.NAME,
        include = JsonTypeInfo.As.PROPERTY,
        property = "type" // Se creará un campo "type": "MAN_MADE_DAM" en el JSON
)
// 2. Registramos las subclases conocidas
@JsonSubTypes({
        @JsonSubTypes.Type(value = GeoEvManMadeDam.class, name = "MAN_MADE_DAM")
        // @JsonSubTypes.Type(value = GeoEvPollution.class, name = "CANON") en el futuro y etc
})
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
            float spatialResolution,
            float[] elevationProfile,
            float[] bottomWidth,
            float[] manningCoefficient,
            RiverSectionType[] sectionTypes
    );
    double getPosition();
}