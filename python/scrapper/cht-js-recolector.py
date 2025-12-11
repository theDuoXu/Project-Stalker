import requests
import json
import os
import re
import time

# --- CONFIGURACIÓN ---
INPUT_FILE = 'saica_stations_master_list.json'
# Directorio base donde se guardarán los batches
BASE_OUTPUT_DIR = '/home/duo/Projects/proyecto-computacion-I/data/datasets/raw'
BASE_URL = "https://saihtajo.chtajo.es/"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Referer': 'https://saihtajo.chtajo.es/'
}
REQUEST_DELAY_SECONDS = 0.2


# --- FUNCIONES AUXILIARES ---

def get_next_batch_dir(base_path):
    """
    Busca carpetas que empiecen por 'batch' seguido de un número,
    encuentra el número más alto y devuelve la ruta para el siguiente batch.
    """
    # Si el directorio base no existe, lo creamos
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        print(f"Directorio base creado: {base_path}")
        return os.path.join(base_path, 'batch1')

    # Listar subdirectorios existentes
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    # Filtrar solo los que siguen el patrón 'batchN'
    batch_numbers = []
    for d in subdirs:
        match = re.match(r'^batch(\d+)$', d)
        if match:
            batch_numbers.append(int(match.group(1)))

    # Determinar el siguiente número
    if not batch_numbers:
        next_num = 1
    else:
        next_num = max(batch_numbers) + 1

    return os.path.join(base_path, f'batch{next_num}')


# --- SCRIPT PRINCIPAL ---

# 1. Determinar y crear el directorio de salida (batch)
output_dir = get_next_batch_dir(BASE_OUTPUT_DIR)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"📂 Carpeta de destino creada: {output_dir}")
else:
    print(f"📂 Usando carpeta de destino: {output_dir}")

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

                    # Guardar los datos en la carpeta del batch correspondiente
                    filename = f"{station_code}_{metric_name.replace(' ', '_')}.json"
                    output_filename = os.path.join(output_dir, filename)

                    with open(output_filename, 'w', encoding='utf-8') as f:
                        json.dump(data_response.json(), f, ensure_ascii=False, indent=4)
                    print(f" ✅ Guardado")
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
print(f"Los ficheros se han guardado en: '{output_dir}'")