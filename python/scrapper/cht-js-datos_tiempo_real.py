import requests
import json

# Esta es la llamada para cargar el contenido de la pestaña "Datos en Tiempo Real".
data_index_url = "https://saihtajo.chtajo.es/index.php?w=get-datos-tiempo-real&x=/ikP2Tzu5snbTbvdgFM6UPHznWfZJNqv//htQaodWed9MOiOUMSrxG3IfBAf8XkShYmEufaflrhH5rVxmTy+5MySySeMsbGEXfsf9VvEDoyiS3wMAWIqrRLZHnPb80FO"

print(f"1. Realizando la llamada a la API para obtener el índice completo de estaciones...")

try:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': 'https://saihtajo.chtajo.es/' # Añadimos un Referer para parecer más legítimos
    }
    response = requests.get(data_index_url, headers=headers)
    response.raise_for_status()

    # La respuesta debería ser el JSON grande con la estructura anidada.
    data_index = response.json()

    print("✅ ¡Éxito! Índice de datos recibido.")

    # Guardamos este JSON.
    output_filename = 'station_index_response.json'
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(data_index, f, ensure_ascii=False, indent=4)
    print(f"   -> La respuesta completa se ha guardado en '{output_filename}'")

except requests.exceptions.RequestException as e:
    print(f"❌ Error al realizar la petición: {e}")
    exit()


print(f"\n2. Explora el fichero '{output_filename}'.")
print("   Dentro de este JSON, encontrarás la jerarquía completa:")
print("   'cuenca' -> 'subcuencas' -> 'tipos' -> 'estaciones' -> 'url'")
print("\n   El siguiente paso es escribir un script que:")
print("   1. Parsee este fichero para extraer la 'url' de cada estación.")
print("   2. Haga una petición a la URL de cada estación para obtener sus métricas ('senales').")
print("   3. De la respuesta de cada señal, extraiga la 'url' del gráfico.")
print("   4. Haga una petición a la URL del gráfico y busque el enlace de oro: 'urlexportarjson'.")
print("   5. Descargue los datos desde el enlace JSON.")