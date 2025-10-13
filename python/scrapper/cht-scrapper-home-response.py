import requests
import json

# La URL de la página "home", que es la que contiene la lista de estaciones y el mapa.
# Las aplicaciones modernas a menudo usan URLs "limpias" como esta.
home_data_url = "https://saihtajo.chtajo.es/home"

print("1. Realizando la llamada a 'home' para obtener los datos de las estaciones...")

try:
    # Hacemos la petición GET. Seguimos usando un User-Agent.
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(home_data_url, headers=headers)
    response.raise_for_status()

    # La respuesta debería ser el JSON grande con todos los datos.
    home_data = response.json()

    print("✅ ¡Éxito! Datos de la página principal recibidos.")

    # Guardamos la respuesta completa para analizarla.
    with open('home_response.json', 'w', encoding='utf-8') as f:
        json.dump(home_data, f, ensure_ascii=False, indent=4)
    print("   -> La respuesta completa se ha guardado en 'home_response.json'")

except requests.exceptions.RequestException as e:
    print(f"❌ Error al realizar la petición: {e}")
    # Si la URL limpia no funciona, probamos la versión con parámetros
    print("   -> Intentando con la URL alternativa: index.php?w=home")
    try:
        alternative_url = "https://saihtajo.chtajo.es/index.php?w=home"
        response = requests.get(alternative_url, headers=headers)
        response.raise_for_status()
        home_data = response.json()
        print("✅ ¡Éxito con la URL alternativa!")
        with open('home_response.json', 'w', encoding='utf-8') as f:
            json.dump(home_data, f, ensure_ascii=False, indent=4)
        print("   -> La respuesta completa se ha guardado en 'home_response.json'")
    except requests.exceptions.RequestException as e_alt:
        print(f"❌ La URL alternativa también ha fallado: {e_alt}")
        exit()

# --- Ahora exploramos ESTE NUEVO JSON ---

print("\n2. Explorando 'home_response.json' en busca de las URLs de los gráficos...")

# Ahora sí, este JSON debería ser mucho más grande y contener la estructura que esperamos.
# Busca claves como 'cuenca', 'subcuencas', 'tipos', 'estaciones', 'senales', etc.
try:
    # Buscamos la primera URL de gráfico que encontremos para hacer una prueba.
    first_signal_url = None

    # La ruta exacta puede variar. Ajusta esto según lo que veas en 'home_response.json'
    subcuencas = home_data.get('response', {}).get('cuenca', {}).get('subcuencas', [])
    for subcuenca in subcuencas:
        if first_signal_url: break
        for tipo in subcuenca.get('tipos', []):
            if first_signal_url: break
            for estacion in tipo.get('estaciones', []):
                # Hacemos una petición a la URL de la estación para obtener sus métricas ('senales')
                estacion_url = f"https://saihtajo.chtajo.es/{estacion.get('url')}"
                estacion_response = requests.get(estacion_url, headers=headers).json()

                for senal in estacion_response.get('response', {}).get('senales', []):
                    # ¡Bingo! Esta es la URL de un gráfico específico.
                    first_signal_url = senal.get('url')
                    if first_signal_url:
                        print(f"   -> URL de gráfico encontrada: {first_signal_url}")
                        break
                if first_signal_url: break

    if not first_signal_url:
        print("   -> No se encontraron URLs de gráficos en la ruta esperada. Revisa 'home_response.json' manualmente.")

except (KeyError, TypeError, AttributeError) as e:
    print(f"❌ Error al navegar el JSON de 'home': {e}")
    print("   -> Abre 'home_response.json' para ver su estructura y ajustar el script.")