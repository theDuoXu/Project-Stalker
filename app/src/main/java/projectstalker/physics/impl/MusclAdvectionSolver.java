package projectstalker.physics.impl;

import projectstalker.domain.river.RiverGeometry;
import projectstalker.physics.i.IAdvectionSolver;

/**
 * Solver de Advección de Alta Resolución (High-Resolution Scheme).
 * Implementa el esquema MUSCL (Monotonic Upstream-Centered Scheme for Conservation Laws)
 * con un limitador de flujo MinMod para garantizar la propiedad TVD (Total Variation Diminishing).
 * Esto permite transportar "ondas cuadradas" (vertidos bruscos) sin que se difuminen
 * demasiado (como pasa con Upwind 1er orden) ni oscilen (como pasa con Lax-Wendroff).
 */
public class MusclAdvectionSolver implements IAdvectionSolver {

    @Override
    public String getName() {
        return "MUSCL_MinMod";
    }

    @Override
    public String getDescription() {
        return "FVM Conservativo de 2do orden con reconstrucción lineal y limitador MinMod.";
    }

    /**
     * Resuelve la ecuación de advección: d(AC)/dt + d(QC)/dx = 0
     */
    @Override
    public float[] solveAdvection(float[] concentration, float[] velocity, float[] area, RiverGeometry geometry, float dt) {
        int n = concentration.length;
        float[] newConcentration = new float[n];
        double dx = geometry.getSpatial_resolution();

        // Array temporal para almacenar los flujos en las caras (i+1/2)
        // flux[i] representa el flujo entre la celda i y la i+1
        float[] flux = new float[n + 1];

        // --- PASO 1: Calcular Flujos en las caras (Reconstrucción + Riemann) ---
        // Iteramos sobre las caras entre celdas (desde 0 hasta N-1)
        // La cara i está entre la celda i y la i+1
        for (int i = 0; i < n - 1; i++) {
            // Necesitamos 4 puntos para MUSCL: i-1, i, i+1, i+2
            // Para simplificar en los bordes, usamos un esquema de bajo orden (Upwind)
            // si estamos muy cerca del inicio o del final.
            if (i == 0 || i >= n - 2) {
                // Borde: Usamos Upwind simple (1er orden)
                // Flujo = Q * C_upwind
                double Q = velocity[i] * area[i]; // Caudal aproximado en la cara
                flux[i] = (float) (Q * concentration[i]);
            } else {
                // Interior: Usamos MUSCL (2do orden)
                flux[i] = calculateMusclFlux(concentration, velocity, area, i, dt, dx);
            }
        }

        // Condiciones de Frontera para Flujos
        // Flujo entrante (i=-1): Asumimos 0 (o valor de frontera Dirichlet si lo hubiera)
        float fluxIn = 0.0f;
        // Flujo saliente (i=N-1): Calculado en el bucle anterior (flux[n-2]) o extrapolado
        // Para la última celda, usamos Upwind simple para sacar el agua
        double Q_last = velocity[n - 1] * area[n - 1];
        flux[n - 1] = (float) (Q_last * concentration[n - 1]); // Flujo que sale del sistema

        // --- PASO 2: Actualizar Estado (Balance de Masas) ---
        // C_new = C_old - (dt / (A * dx)) * (Flux_out - Flux_in)
        for (int i = 0; i < n; i++) {
            float fluxOutRight = flux[i];     // Flujo que sale por la derecha (cara i)
            float fluxInLeft = (i == 0) ? fluxIn : flux[i - 1]; // Flujo que entra por la izquierda

            // Volumen de control V = A * dx
            double volume = area[i] * dx;

            // Evitar división por cero si el río está seco
            if (volume < 1e-9) {
                newConcentration[i] = 0.0f;
            } else {
                double change = (dt / volume) * (fluxInLeft - fluxOutRight);
                newConcentration[i] = (float) (concentration[i] + change);
            }

            // Saneamiento: La concentración no puede ser negativa por errores numéricos
            if (newConcentration[i] < 0) newConcentration[i] = 0.0f;
        }

        return newConcentration;
    }

    /**
     * Calcula el flujo numérico en la cara entre i e i+1 usando reconstrucción MUSCL.
     */
    private float calculateMusclFlux(float[] c, float[] u, float[] A, int i, float dt, double dx) {
        // Índices vecinos
        int prev = i - 1;
        int curr = i;
        int next = i + 1;
        int next2 = i + 2;

        // 1. Calcular pendientes (r)
        // r_i = (C_i - C_i-1) / (C_i+1 - C_i)
        // Mide qué tan suave es el cambio. Si r cerca de 1, es suave. Si r lejos, es un pico.
        float slope_left = c[curr] - c[prev];
        float slope_right = c[next] - c[curr];

        // 2. Aplicar Limitador MinMod para obtener la pendiente limitada (phi)
        // Esto evita oscilaciones. Si las pendientes son opuestas (pico), la pendiente es 0 (plana).
        float limitedSlope = minmod(slope_left, slope_right);

        // 3. Reconstrucción de valores en la cara (i + 1/2)
        // Extrapolamos desde el centro de la celda 'i' hasta su borde derecho.
        // C_L = C_i + 0.5 * phi * (C_i+1 - C_i)  <-- Simplificación estándar MUSCL
        // Nota: Para MUSCL-Hancock completo se usa dt también, aquí usamos la versión espacial
        float c_left_at_face = c[curr] + 0.5f * limitedSlope;

        // 4. Calcular Caudal en la cara (Promedio o Upwind)
        // Como v > 0 siempre (tu premisa), el flujo viene de la izquierda.
        // Usamos la velocidad y área de la celda 'i' (Upwind).
        double Q = u[curr] * A[curr];

        // 5. Flujo Final
        return (float) (Q * c_left_at_face);
    }

    /**
     * Función Limitadora MinMod.
     * Devuelve el valor más pequeño en magnitud si ambos tienen el mismo signo.
     * Devuelve 0 si tienen signos opuestos (es un máximo o mínimo local).
     */
    private float minmod(float a, float b) {
        if (a * b <= 0) return 0.0f; // Signos opuestos -> 0
        if (Math.abs(a) < Math.abs(b)) return a;
        return b;
    }
}