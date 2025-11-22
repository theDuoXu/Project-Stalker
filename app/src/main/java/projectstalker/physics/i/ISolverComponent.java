package projectstalker.physics.i;

/**
 * Contrato base para cualquier componente numérico del sistema.
 * Permite tratar a todos los solvers de forma polimórfica para tareas
 * de logging, identificación y depuración, sin importar su física.
 */
public interface ISolverComponent {
    /**
     * Nombre corto del algoritmo (ej: "MUSCL-Hancock", "Crank-Nicolson").
     */
    String getName();

    /**
     * Descripción técnica detallada (ej: "Esquema de 2do orden TVD con limitador MinMod").
     */
    default String getDescription() {
        return "Sin descripción disponible.";
    }
}
