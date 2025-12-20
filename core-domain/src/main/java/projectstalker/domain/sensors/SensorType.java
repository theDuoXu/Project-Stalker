package projectstalker.domain.sensors;

import lombok.Getter;
import lombok.RequiredArgsConstructor;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Getter
@RequiredArgsConstructor
public enum SensorType {

    TEMPERATURA("TEMPERATURA", "ºC", "DIGITAL"),
    NIVEL("NIVEL", "m", "ANALOG"),
    CONDUCTIVIDAD("CONDUCTIVIDAD", "µS/cm", "DIGITAL"),
    PH("PH", "pH", "DIGITAL"),
    OXIGENO_DISUELTO("OXIGENO_DISUELTO", "mg/l", "OPTICAL"),

    CLOROFILA("CLOROFILA", "µg/l", "FLUORESCENCE"),
    FICOCIANINAS("FICOCIANINAS", "µg/l", "FLUORESCENCE"),

    TURBIDEZ("TURBIDEZ", "NTU", "OPTICAL"),
    AMONIO("AMONIO", "mg/l", "ISE"),
    NITRATOS("NITRATOS", "mg/l", "UV"),
    FOSFATOS("FOSFATOS", "mg/l", "COLORIMETRIC"),
    CARBONO_ORGANICO("CARBONO_ORGANICO", "mg/l", "UV"),

    // --- FALLBACK ---
    UNKNOWN("UNKNOWN", "-", "UNKNOWN");

    private final String code;
    private final String unit;
    private final String signalType;

    // 1. MAPA DE BÚSQUEDA POR CÓDIGO
    // Clave: CÓDIGO (Único) -> Valor: ENUM
    private static final Map<String, SensorType> BY_CODE = Collections.unmodifiableMap(
            Arrays.stream(values())
                    .collect(Collectors.toMap(
                            type -> type.code.toUpperCase(), // Clave normalizada
                            type -> type
                    ))
    );

    // 2. MAPA DE BÚSQUEDA POR UNIDAD
    // Clave: UNIDAD -> Valor: ENUM
    private static final Map<String, SensorType> BY_UNIT = Collections.unmodifiableMap(
            Arrays.stream(values())
                    .collect(Collectors.toMap(
                            type -> type.unit.toLowerCase(),
                            type -> type,
                            // FUNCIÓN DE FUSIÓN (Merge Function):
                            // Si hay colisión (ej: dos tienen "µg/l"), nos quedamos con el primero (existing).
                            // Esto evita la excepción "Duplicate key".
                            (existing, replacement) -> existing
                    ))
    );

    /**
     * Busca un SensorType por su código de forma eficiente (O(1)).
     */
    public static SensorType fromString(String text) {
        if (text == null) return UNKNOWN;
        // Búsqueda directa en Hash Map
        return BY_CODE.getOrDefault(text.trim().toUpperCase(), UNKNOWN);
    }

    /**
     * Devuelve el primer SensorType asociado a la unidad dada.
     * Nota: Como varias sondas pueden compartir unidad, esto es ambiguo y devuelve la primera coincidencia.
     */
    public static SensorType fromUnit(String unit) {
        if (unit == null || unit.trim().isEmpty()) {
            return UNKNOWN;
        }
        return BY_UNIT.getOrDefault(unit.trim().toLowerCase(), UNKNOWN);
    }


    /**
     * Devuelve una etiqueta amigable para comboboxes: "Nombre (Unidad)"
     */
    public String getFriendlyName() {
        // Convierte "OXIGENO_DISUELTO" a "Oxigeno Disuelto"
        String name = code.replace("_", " ");
        name = name.substring(0, 1).toUpperCase() + name.substring(1).toLowerCase();
        return String.format("%s (%s)", name, unit);
    }

    /**
     * Devuelve la lista de todos los tipos válidos (excluyendo UNKNOWN)
     */
    public static List<SensorType> getValidTypes() {
        return Arrays.stream(values())
                .filter(t -> t != UNKNOWN)
                .collect(Collectors.toList());
    }
}