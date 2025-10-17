package projectstalker.physics.model;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import projectstalker.domain.river.RiverGeometry;
import projectstalker.config.RiverConfig;
import projectstalker.factory.RiverGeometryFactory;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

/**
 * Test unitario para RiverPhModel.
 * Se ha corregido el uso del mock para garantizar que clonePhProfile()
 * simule la devolución de una COPIA (array nuevo) en cada llamada.
 */
@Slf4j
class RiverPhModelTest {

    private RiverGeometry mockGeometry;
    private RiverPhModel phModel;
    private final double[] BASE_PH_PROFILE = new double[]{7.0, 7.1, 7.2, 7.3, 7.4};

    @BeforeEach
    void setUp() {
        // Inicializar el mock de RiverGeometry
        mockGeometry = mock(RiverGeometry.class);
        when(mockGeometry.clonePhProfile()).thenAnswer(invocation -> BASE_PH_PROFILE.clone());

        // Simular el tamaño de la geometría
        when(mockGeometry.getCellCount()).thenReturn(BASE_PH_PROFILE.length);

        // Inicializar la clase a testear
        phModel = new RiverPhModel(mockGeometry);
        log.info("Test inicializado con RiverGeometry mock que garantiza la clonación del perfil de pH.");
    }

    @Test
    @DisplayName("El perfil de pH debe ser devuelto como una copia para evitar mutaciones externas")
    void getPhProfile_shouldReturnClonedProfile() {
        log.info("Ejecutando test: getPhProfile_shouldReturnClonedProfile.");

        // ARRANGE
        // Obtenemos el primer perfil
        double[] phProfile = phModel.getPhProfile();

        // ASSERT 1: Verificar el valor inicial antes de la mutación
        assertEquals(BASE_PH_PROFILE[0], phProfile[0], "El valor inicial del pH debe ser 7.0.");

        // ACT (Mutación)
        log.warn("Mutando la copia devuelta: phProfile[0] = 99.0");
        phProfile[0] = 99.0;

        // ARRANGE (Obtener un segundo perfil, que DEBE ser una COPIA nueva del original)
        double[] secondPhProfile = phModel.getPhProfile();

        // ASSERT 2: Verificar que el segundo perfil obtenido NO fue afectado por la mutación
        // ESTA ES LA VERIFICACIÓN CLAVE DEL REQUISITO DE CLONACIÓN/INMUTABILIDAD
        assertEquals(BASE_PH_PROFILE[0], secondPhProfile[0], 0.001,
                "El array devuelto debe ser una copia. La mutación a 99.0 no debe afectar la segunda llamada.");

        // Verificar que el método delegado fue llamado dos veces
        verify(mockGeometry, times(2)).clonePhProfile();

        log.info("Perfil de pH verificado exitosamente. La mutación externa fue aislada.");
    }

    @Test
    @DisplayName("Verificación de valores y tamaño del perfil")
    void getPhProfile_shouldReturnCorrectValuesAndSize() {
        // Act
        double[] phProfile = phModel.getPhProfile();

        // Assert
        assertNotNull(phProfile);
        assertEquals(BASE_PH_PROFILE.length, phProfile.length, "El tamaño del perfil debe ser 5.");
        assertArrayEquals(BASE_PH_PROFILE, phProfile, 0.001, "Los valores deben coincidir con el perfil base.");
        log.info("Valores y tamaño del perfil verificados.");
    }
}