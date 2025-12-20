package projectstalker.ui.security;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.awt.Desktop;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.SecureRandom;
import java.util.Base64;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.CompletableFuture;

@Service
public class AuthenticationService {

    private final WebClient webClient;
    private final String authServerUrl;
    private final String realm;
    private final String clientId;

    // Guardamos el token en memoria
    private TokenResponse currentToken;

    // Instancia de Jackson para leer el JSON del token
    private final ObjectMapper objectMapper = new ObjectMapper();

    public AuthenticationService(
            WebClient.Builder webClientBuilder,
            @Value("${app.security.auth-server-url}") String authServerUrl,
            @Value("${app.security.realm}") String realm,
            @Value("${app.security.client-id}") String clientId) {

        this.webClient = webClientBuilder.build();
        this.authServerUrl = authServerUrl;
        this.realm = realm;
        this.clientId = clientId;
    }

    /**
     * Inicia el flujo "Authorization Code" con PKCE.
     * 1. Levanta un socket temporal.
     * 2. Abre el navegador.
     * 3. Espera el código.
     * 4. Canjea el código por token.
     */
    public CompletableFuture<TokenResponse> login() {
        return CompletableFuture.supplyAsync(() -> {
            try {
                // 1. Preparar PKCE
                String codeVerifier = generateCodeVerifier();
                String codeChallenge = generateCodeChallenge(codeVerifier);

                // 2. Buscar puerto libre y preparar listener
                ServerSocket serverSocket = new ServerSocket(0); // Puerto 0 = aleatorio libre
                int port = serverSocket.getLocalPort();
                String redirectUri = "http://localhost:" + port + "/callback";

                // 3. Construir URL de Login
                String loginUrl = String.format(
                        "%s/realms/%s/protocol/openid-connect/auth" +
                                "?response_type=code" +
                                "&client_id=%s" +
                                "&redirect_uri=%s" +
                                "&scope=openid%%20profile%%20email" +
                                "&code_challenge=%s" +
                                "&code_challenge_method=S256",
                        authServerUrl, realm, clientId, redirectUri, codeChallenge
                );

                // 4. Abrir navegador del sistema
                if (Desktop.isDesktopSupported() && Desktop.getDesktop().isSupported(Desktop.Action.BROWSE)) {
                    Desktop.getDesktop().browse(new URI(loginUrl));
                } else {
                    throw new RuntimeException("No se puede abrir el navegador del sistema.");
                }

                // 5. Esperar el callback (Bloqueante hasta que el usuario se loguea en el navegador)
                String authCode = waitForCallback(serverSocket);

                // 6. Intercambiar el código por el token (Petición POST directa)
                return exchangeCodeForToken(authCode, redirectUri, codeVerifier);

            } catch (Exception e) {
                throw new RuntimeException("Error en el flujo de login: " + e.getMessage(), e);
            }
        });
    }

    /**
     * Cierra la sesión local y abre el navegador para cerrar la sesión en Keycloak.
     * Se usa id_token_hint para evitar que Keycloak pregunte "¿Estás seguro?".
     */
    public CompletableFuture<Void> logout() {
        return CompletableFuture.runAsync(() -> {
            try {
                if (currentToken == null) {
                    return;
                }

                // 1. Construir URL de Logout
                // Endpoint estándar: /protocol/openid-connect/logout
                StringBuilder logoutUrl = new StringBuilder();
                logoutUrl.append(String.format("%s/realms/%s/protocol/openid-connect/logout", authServerUrl, realm));

                // Parámetros necesarios para evitar el prompt de confirmación
                logoutUrl.append("?client_id=").append(clientId);

                 String idToken = currentToken.idToken();
                 if (idToken != null) {
                    logoutUrl.append("&id_token_hint=").append(idToken);
                 }

                // 2. Abrir navegador para limpiar cookies de sesión (Front-channel logout)
                if (Desktop.isDesktopSupported() && Desktop.getDesktop().isSupported(Desktop.Action.BROWSE)) {
                    Desktop.getDesktop().browse(new URI(logoutUrl.toString()));
                }

            } catch (Exception e) {
                // Solo logueamos, no detenemos el logout local
                System.err.println("Error abriendo navegador para logout: " + e.getMessage());
            } finally {
                // 3. Limpieza local
                this.currentToken = null;
            }
        });
    }

    /**
     * Servidor HTTP "de usar y tirar" para capturar el código de la URL.
     */
    private String waitForCallback(ServerSocket serverSocket) throws IOException {
        try (serverSocket; Socket clientSocket = serverSocket.accept()) {
            // Leer la petición HTTP del navegador
            BufferedReader in = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()));
            String line = in.readLine(); // Ej: GET /callback?code=XXXX... HTTP/1.1

            // Responder al navegador para que el usuario sepa que puede cerrar
            OutputStream out = clientSocket.getOutputStream();
            String successUrl = "https://auth.protonenergyindustries.com/login-success";
            String httpResponse = "HTTP/1.1 302 Found\r\n" +
                    "Location: " + successUrl + "\r\n" +
                    "Connection: close\r\n" +
                    "\r\n";
            out.write(httpResponse.getBytes(StandardCharsets.UTF_8));
            out.flush();

            // Parsear el código de la primera línea
            if (line != null && line.contains("code=")) {
                String[] parts = line.split("code=");
                if (parts.length > 1) {
                    // El código termina con un espacio (HTTP/1.1)
                    return parts[1].split(" ")[0];
                }
            }
            throw new RuntimeException("No se recibió el código de autorización.");
        }
    }

    private TokenResponse exchangeCodeForToken(String code, String redirectUri, String codeVerifier) {
        String tokenUrl = String.format("%s/realms/%s/protocol/openid-connect/token", authServerUrl, realm);

        MultiValueMap<String, String> formData = new LinkedMultiValueMap<>();
        formData.add("grant_type", "authorization_code");
        formData.add("client_id", clientId);
        formData.add("code", code);
        formData.add("redirect_uri", redirectUri);
        formData.add("code_verifier", codeVerifier);

        TokenResponse response = webClient.post()
                .uri(tokenUrl)
                .contentType(MediaType.APPLICATION_FORM_URLENCODED)
                .body(BodyInserters.fromFormData(formData))
                .retrieve()
                .bodyToMono(TokenResponse.class)
                .block(); // Bloqueamos aquí porque ya estamos en un hilo asíncrono (CompletableFuture)

        this.currentToken = response;
        return response;
    }

    // --- Utilidades PKCE ---

    private String generateCodeVerifier() {
        SecureRandom sr = new SecureRandom();
        byte[] code = new byte[32];
        sr.nextBytes(code);
        return Base64.getUrlEncoder().withoutPadding().encodeToString(code);
    }

    private String generateCodeChallenge(String codeVerifier) throws Exception {
        byte[] bytes = codeVerifier.getBytes(StandardCharsets.US_ASCII);
        MessageDigest md = MessageDigest.getInstance("SHA-256");
        md.update(bytes, 0, bytes.length);
        byte[] digest = md.digest();
        return Base64.getUrlEncoder().withoutPadding().encodeToString(digest);
    }

    public String getAccessToken() {
        return currentToken != null ? currentToken.accessToken() : null;
    }

    public boolean isAuthenticated() {
        return currentToken != null;
    }


    /**
     * Decodifica el Access Token (JWT) y extrae los roles.
     * Busca en 'realm_access' y 'resource_access' (client roles).
     */
    public Set<String> getCurrentRoles() {
        if (currentToken == null || currentToken.accessToken() == null) {
            return Collections.emptySet();
        }

        try {
            // 1. El JWT tiene 3 partes: Header.Payload.Signature
            // Nos interesa el Payload
            String[] chunks = currentToken.accessToken().split("\\.");
            if (chunks.length < 2) {
                return Collections.emptySet();
            }

            // 2. Decodificar Base64Url
            Base64.Decoder decoder = Base64.getUrlDecoder();
            String payload = new String(decoder.decode(chunks[1]), StandardCharsets.UTF_8);

            // 3. Parsear el JSON
            JsonNode rootNode = objectMapper.readTree(payload);
            Set<String> roles = new HashSet<>();

            // 4. Extraer REALM ROLES (realm_access.roles)
            // Estructura: "realm_access": { "roles": ["ADMIN", "USER"] }
            JsonNode realmAccess = rootNode.path("realm_access");
            if (!realmAccess.isMissingNode() && realmAccess.has("roles")) {
                realmAccess.get("roles").forEach(roleNode -> roles.add(roleNode.asText()));
            }

            // 5. Extraer CLIENT ROLES (resource_access.clientId.roles)
            // Estructura: "resource_access": { "my-client": { "roles": [...] } }
            // Apunte, solo hay realm roles por ahora pero ya lo dejamos listo para el futuro
            JsonNode resourceAccess = rootNode.path("resource_access").path(clientId);
            if (!resourceAccess.isMissingNode() && resourceAccess.has("roles")) {
                resourceAccess.get("roles").forEach(roleNode -> roles.add(roleNode.asText()));
            }

            return roles;

        } catch (Exception e) {
            System.err.println("Error decodificando roles del token: " + e.getMessage());
            return Collections.emptySet();
        }
    }
}