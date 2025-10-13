import requests
import json

# La URL de arranque que encontramos al final del HTML.
# Esta es la primera llamada que hace la web para cargar toda su estructura.
startup_url = "https://saihtajo.chtajo.es/index.php?w=get-wrapperentorno&x=XJrnqUE2vLYwSoH5d19QbbXNE%2BO8z3mda%2Fj8jSIHXuJJUvI41UwTMLHtmJWfmip33fsShcuw57NIrcRBe2HXWQ%3D%3D"

print("1. Realizando la llamada de arranque para obtener la estructura del sitio...")

try:
    # Hacemos la petición GET. Usamos un User-Agent para parecer un navegador normal.
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(startup_url, headers=headers)

    # Esto lanzará un error si la petición falla (ej. error 404 o 500)
    response.raise_for_status()

    # La respuesta es un JSON. Lo parseamos a un diccionario de Python.
    initial_data = response.json()

    print("✅ ¡Éxito! Respuesta inicial recibida y parseada.")

    # (Opcional) Guardamos la respuesta completa en un fichero para analizarla con calma.
    with open('initial_response.json', 'w', encoding='utf-8') as f:
        json.dump(initial_data, f, ensure_ascii=False, indent=4)
    print("   -> La respuesta completa se ha guardado en 'initial_response.json'")

except requests.exceptions.RequestException as e:
    print(f"❌ Error al realizar la petición: {e}")
    exit()

# --- A partir de aquí, empezamos a explorar los datos ---

print("\n2. Buscando las URLs de los datos históricos...")

# Navegamos la estructura del JSON para encontrar las URLs encriptadas.
# La estructura exacta puede variar, tendrás que explorar 'initial_response.json'
# para encontrar la ruta correcta. Asumamos que está en una clave como 'cuenca' -> 'subcuencas', etc.

data_urls = []
try:
    # Este es un ejemplo conceptual de cómo podrías navegar la estructura.
    # Necesitarás ajustar las claves ('cuenca', 'subcuencas', etc.) a lo que veas en el JSON.
    subcuencas = initial_data.get('response', {}).get('cuenca', {}).get('subcuencas', [])
    for subcuenca in subcuencas:
        for tipo in subcuenca.get('tipos', []):
            for estacion in tipo.get('estaciones', []):
                # Aquí encontramos la URL para obtener los datos de una estación completa
                estacion_url = estacion.get('url')
                if estacion_url:
                    # NOTA: La URL podría ser relativa, así que la completamos.
                    full_estacion_url = f"https://saihtajo.chtajo.es/{estacion_url}"
                    # Aquí harías otra petición a esta URL para obtener las métricas de la estación.
                    # Por simplicidad, lo dejamos para el siguiente paso.

# Ahora, nos centramos en las URLs que ya tienes para los gráficos:
# La lógica real será iterar sobre las estaciones, hacer una petición para cada una
# y luego, en esa respuesta, buscar las URLs de los gráficos.

# El siguiente paso sería hacer una petición a una de esas URLs (las que contienen 'get-estacion-grafico-grande')
# y buscar la clave de oro: "urlexportarjson".

except (KeyError, TypeError) as e:
    print(f"❌ No se pudo encontrar la ruta esperada en el JSON: {e}")
    print("   -> Revisa el fichero 'initial_response.json' para entender la estructura correcta.")

print("\n✅ Proceso finalizado. El siguiente paso es iterar sobre las URLs de las estaciones,")
print("   hacer las peticiones y extraer los enlaces 'urlexportarjson' para la descarga final.")