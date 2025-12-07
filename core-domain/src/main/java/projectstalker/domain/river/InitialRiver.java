// Puedes poner este record en el paquete 'projectstalker.domain.river'
package projectstalker.domain.river;

/**
 * Contenedor para un par de Geometría y Estado del río,
 * representando un sistema fluvial completo en un instante.
 */
public record InitialRiver(RiverGeometry geometry, RiverState state) {}