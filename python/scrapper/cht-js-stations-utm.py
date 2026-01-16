import json
import requests
import time
import os

# --- CONFIGURACIÓN ---
INPUT_FILE = 'saica_stations_master_list.json'
OUTPUT_FILE = '../../data/datasets/raw/saica_stations_maestro_coords.json'

# Cabeceras para simular un navegador real y evitar bloqueos
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Referer': 'https://saihtajo.chtajo.es/'
}
REQUEST_DELAY = 0.5 # Medio segundo de cortesía entre peticiones

def enriquecer_sensores():
    # 1. Cargar el maestro original
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: No encuentro {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        stations = json.load(f)

    print(f"📡 Iniciando enriquecimiento de {len(stations)} estaciones...")

    stations_with_coords = []

    # 2. Iterar y extraer coordenadas
    for i, station in enumerate(stations):
        print(f"[{i+1}/{len(stations)}] Procesando {station['codigo']}...", end=" ")

        try:
            # Hacemos la petición a la URL que descubriste en el XHR
            resp = requests.get(station['url_detalles'], headers=HEADERS, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            # Navegamos el JSON de respuesta para sacar las UTM
            # Estructura observada: response -> utm -> {x, y, huso}
            utm_data = data.get('response', {}).get('utm', {})

            if utm_data:
                # Inyectamos las coordenadas en nuestro objeto estación
                station['utm_x'] = utm_data.get('x')
                station['utm_y'] = utm_data.get('y')
                station['huso'] = utm_data.get('huso')
                print(f"✅ Coordenadas obtenidas (X: {station['utm_x']})")
            else:
                print("⚠️ No se encontraron datos UTM en la respuesta")

        except Exception as e:
            print(f"❌ Error conectando: {e}")

        # Guardamos la estación (con o sin coords) para no perderla
        stations_with_coords.append(station)
        time.sleep(REQUEST_DELAY)

    # 3. Guardar el nuevo maestro enriquecido
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(stations_with_coords, f, ensure_ascii=False, indent=4)

    print(f"\n🎉 ¡Misión cumplida! Archivo generado: {OUTPUT_FILE}")
    print("Listo para importar en Neo4j.")

if __name__ == "__main__":
    enriquecer_sensores()