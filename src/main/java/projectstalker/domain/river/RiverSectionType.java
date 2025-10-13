package projectstalker.domain.river;

public enum RiverSectionType {
    NATURAL,         // Ningún evento geológico importante
    LANDSLIDE,      // Deslizamiento de tierra que eleva el lecho.
    TECTONIC_UPLIFT, // Levantamiento de una sección del terreno.
    SUBSIDENCE,     // Hundimiento de una sección del terreno.
    NATURAL_DAM,     // Un deslizamiento masivo que crea una presa natural.
    DAM_STRUCTURE,    // Presa hidroeléctrica
    RESERVOIR
}
