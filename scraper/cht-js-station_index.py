import json

# Nombre del fichero que contiene el índice completo
index_filename = 'station_index_response.json'
# Fichero donde guardaremos nuestra lista de tareas
output_filename = 'saica_stations_master_list.json'
# Base de la URL para construir las direcciones completas
base_url = "https://saihtajo.chtajo.es/"

print(f"1. Cargando el índice de estaciones desde '{index_filename}'...")

try:
    with open(index_filename, 'r', encoding='utf-8') as f:
        data_index = json.load(f)
except FileNotFoundError:
    print(
        f"❌ Error: No se encuentra el fichero '{index_filename}'. Asegúrate de que está en la misma carpeta que el script.")
    exit()

saica_stations = []
print("2. Extrayendo todas las estaciones de calidad del agua (SAICA)...")

# Navegamos por la estructura que hemos descubierto
try:
    subcuencas = data_index['response']['cuenca']['subcuencas']
    for subcuenca in subcuencas:
        for tipo in subcuenca.get('tipos', []):
            if tipo.get('nombre_limpio') == 'estacion-calidad-saica':
                for estacion in tipo.get('estaciones', []):
                    # Construimos la URL completa
                    full_url = f"{base_url}{estacion.get('url')}"

                    station_info = {
                        "codigo": estacion.get('codigo'),
                        "nombre": estacion.get('nombre'),
                        "subcuenca": subcuenca.get('nombre'),
                        "url_detalles": full_url
                    }
                    saica_stations.append(station_info)

                    print(f"   -> Encontrada: {estacion.get('codigo')} - {estacion.get('nombre')}")

except (KeyError, TypeError) as e:
    print(f"❌ Error al parsear el JSON. La estructura puede haber cambiado. Error: {e}")
    exit()

print(f"\n✅ Proceso completado. Se han encontrado {len(saica_stations)} estaciones SAICA.")

# Guardamos nuestra lista de trabajo para el siguiente script
with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(saica_stations, f, ensure_ascii=False, indent=4)

print(f"   -> La lista completa de estaciones y sus URLs se ha guardado en '{output_filename}'")
print(
    "\nSiguiente paso: Crear un script que lea este nuevo fichero y haga una petición a cada 'url_detalles' para encontrar los enlaces de descarga.")