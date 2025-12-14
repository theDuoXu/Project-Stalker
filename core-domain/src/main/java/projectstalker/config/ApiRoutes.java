package projectstalker.config;

public final class ApiRoutes {

    private ApiRoutes() {}

    // Versión base
    public static final String CURRENT_VERSION = "/v1";

    // Rutas específicas
    public static final String TWINS = CURRENT_VERSION + "/twins";
    public static final String SENSORS = CURRENT_VERSION + "/sensors";
    public static final String SIMULATIONS = CURRENT_VERSION + "/sims";
    public static final String ADMIN = CURRENT_VERSION + "/admin";
    public static final String ALERTS = CURRENT_VERSION + "/alerts";
    public static final String REPORTS = CURRENT_VERSION + "/reports";

    // Rutas públicas
    public static final String API_DOCS = CURRENT_VERSION + "/api-docs";
    public static final String SWAGGER_UI = CURRENT_VERSION + "/swagger-ui";
    public static final String PUBLIC = CURRENT_VERSION + "/public";
}