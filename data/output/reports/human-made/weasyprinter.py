import sys
import os
from weasyprint import HTML

def generar_pdf():
    input_file = "Memoria_proyecto_r4.0.html"
    output_file = "Memoria_Final.pdf"

    # Verificación simple de existencia
    if not os.path.exists(input_file):
        print(f"Error: No se encuentra el archivo '{input_file}' en esta carpeta.")
        return

    print(f"procesando {input_file}...")
    print("Descargando fuentes y renderizando páginas (esto puede tardar unos segundos)...")

    try:
        # Renderizado directo:
        # base_url='.' asegura que si tienes imágenes locales (./img/foto.png), las encuentre.
        HTML(input_file, base_url='.').write_pdf(output_file)
        
        print(f"¡Éxito! PDF guardado como: {output_file}")
    
    except Exception as e:
        print(f"Ocurrió un error: {e}")

if __name__ == "__main__":
    generar_pdf()
