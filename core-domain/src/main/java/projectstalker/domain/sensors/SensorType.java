package projectstalker.domain.sensors;

import lombok.Getter;
import lombok.RequiredArgsConstructor;

import java.util.Arrays;
import java.util.Collections;
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
    private final String signalType; // Metadato técnico (tipo de sonda)

    // 1. MAPA ESTÁTICO DE BÚSQUEDA POR UNIDAD
    // La clave es la unidad (String) y el valor es el SensorType (Enum)
    private static final Map<String, SensorType> UNIT_MAP = Collections.unmodifiableMap(
            Arrays.stream(values())
                    .collect(Collectors.toMap(
                            // La unidad es la clave. La pasamos a minúsculas para una búsqueda sin distinción de mayúsculas/minúsculas.
                            type -> type.unit.toLowerCase(),
                            // El valor es el propio SensorType
                            type -> type
                    ))
    );

    /**
     * Busca un SensorType por su código ignorando mayúsculas/minúsculas.
     */
    public static SensorType fromString(String text) {
        if (text == null) return UNKNOWN;
        for (SensorType t : values()) {
            if (t.code.equalsIgnoreCase(text.trim())) {
                return t;
            }
        }
        return UNKNOWN;
    }

    /**
     * Devuelve el SensorType asociado a la unidad de medida dada.
     * La búsqueda es eficiente (usa un mapa) e ignora mayúsculas/minúsculas.
     * @param unit La unidad de medida (ej. "mg/l", "ºC").
     * @return El SensorType correspondiente, o UNKNOWN si no se encuentra.
     */
    public static SensorType fromUnit(String unit) {
        if (unit == null || unit.trim().isEmpty()) {
            return UNKNOWN;
        }
        // Usamos el mapa estático para una búsqueda O(1)
        return UNIT_MAP.getOrDefault(unit.trim().toLowerCase(), UNKNOWN);
    }
}