import requests
import json
import os
import time

# --- CONFIGURACIÓN ---
INPUT_FILE = 'saica_stations_master_list.json'
OUTPUT_DIR = 'data'
BASE_URL = "https://saihtajo.chtajo.es/"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Referer': 'https://saihtajo.chtajo.es/'
}
# Pequeña pausa entre peticiones para no saturar el servidor
REQUEST_DELAY_SECONDS = 0.5

# --- SCRIPT ---

# 1. Crear el directorio de salida si no existe
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Directorio '{OUTPUT_DIR}' creado.")

# 2. Cargar la lista de estaciones
try:
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        stations = json.load(f)
    print(f"✅ Cargadas {len(stations)} estaciones desde '{INPUT_FILE}'.")
except FileNotFoundError:
    print(f"❌ Error: No se encuentra el fichero '{INPUT_FILE}'.")
    exit()

print("\n--- Iniciando la recolección de datos históricos ---\n")

# 3. Iterar sobre cada estación de la lista
for i, station in enumerate(stations):
    station_code = station['codigo']
    station_name = station['nombre']
    details_url = station['url_detalles']

    print(f"[{i + 1}/{len(stations)}] Procesando estación: {station_code} - {station_name}")

    try:
        # --- PASO 1: Obtener las métricas ('senales') de la estación ---
        time.sleep(REQUEST_DELAY_SECONDS)
        details_response = requests.get(details_url, headers=HEADERS)
        details_response.raise_for_status()
        station_details = details_response.json()

        senales = station_details.get('response', {}).get('senales', [])
        if not senales:
            print(f"   -> ⚠️ No se encontraron métricas ('senales') para esta estación. Saltando.")
            continue

        print(f"   -> Encontradas {len(senales)} métricas. Obteniendo datos para cada una...")

        # --- PASO 2: Iterar sobre cada métrica para obtener su enlace de descarga ---
        for senal in senales:
            metric_name = senal.get('nombre', 'desconocido')
            metric_url = f"{BASE_URL}{senal.get('url')}"

            try:
                time.sleep(REQUEST_DELAY_SECONDS)
                metric_response = requests.get(metric_url, headers=HEADERS)
                metric_response.raise_for_status()
                metric_details = metric_response.json()

                # --- PASO 3: Encontrar y descargar los datos desde la URL de exportación ---
                export_url = metric_details.get('response', {}).get('urlexportarjson')

                if export_url:
                    full_export_url = f"{BASE_URL}{export_url}"
                    print(f"      -> Descargando '{metric_name}'...", end="")

                    time.sleep(REQUEST_DELAY_SECONDS)
                    data_response = requests.get(full_export_url, headers=HEADERS)
                    data_response.raise_for_status()

                    # Guardar los datos en un fichero
                    output_filename = os.path.join(OUTPUT_DIR, f"{station_code}_{metric_name.replace(' ', '_')}.json")
                    with open(output_filename, 'w', encoding='utf-8') as f:
                        json.dump(data_response.json(), f, ensure_ascii=False, indent=4)
                    print(f" ✅ Guardado en {output_filename}")
                else:
                    print(f"      -> ⚠️ No se encontró 'urlexportarjson' para la métrica '{metric_name}'.")

            except requests.exceptions.RequestException as e:
                print(f"      -> ❌ Error al obtener datos para la métrica '{metric_name}': {e}")
            except json.JSONDecodeError:
                print(f"      -> ❌ Error: La respuesta para '{metric_name}' no es un JSON válido.")

    except requests.exceptions.RequestException as e:
        print(f"   -> ❌ Error al obtener los detalles de la estación {station_code}: {e}")
    except json.JSONDecodeError:
        print(f"   -> ❌ Error: La respuesta para la estación {station_code} no es un JSON válido.")

    print("-" * 20)

print("\n🎉 ¡Proceso de recolección de datos finalizado! 🎉")
print(f"Revisa la carpeta '{OUTPUT_DIR}' para ver los ficheros JSON descargados.")