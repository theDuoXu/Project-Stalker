import argparse
import json
import shutil
from pathlib import Path
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Ingesta: JSONs a Parquet particionado por batch (normalizado + deduplicado)"
    )
    p.add_argument(
        "--raw_dir",
        default="~/Projects/proyecto-computacion-I/data/datasets/raw",
        help="Directorio con batch*/ y JSONs",
    )
    p.add_argument(
        "--output_dir",
        default="~/Projects/proyecto-computacion-I/data/datasets/processed/01_ingested/measurements",
        help="Directorio de salida Parquet particionado por batch",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Si se indica, borra el output_dir antes de regenerar el dataset",
    )
    return p.parse_args()


def process_json(filepath: Path, batch_name: str) -> pd.DataFrame:
    """Función que lee un JSON y devuelve un DataFrame normalizado."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)

    resp = data.get("response", {})
    metric_name = resp.get("nombre", "metrica_desconocida")
    unit = resp.get("unidad", "unidad_desconocida")
    valores = resp.get("valores", [])

    if not valores:
        return pd.DataFrame()

    df = pd.DataFrame(valores)

    # Si falta alguna columna básica, se ignora ese JSON
    if "tiempo" not in df.columns or "valor" not in df.columns:
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(
        df["tiempo"], format="%d/%m/%Y %H:%M", errors="coerce"
    )
    df["value"] = pd.to_numeric(df["valor"], errors="coerce")

    df["metric"] = metric_name
    df["unit"] = unit
    df["batch"] = batch_name
    df["sensor_id"] = filepath.stem.split("_")[0]

    cols_to_keep = ["timestamp", "value", "sensor_id", "metric", "unit", "batch"]
    return df[cols_to_keep]


def main():
    args = parse_args()

    RAW_DIR = Path(args.raw_dir).expanduser()
    OUTPUT_DIR = Path(args.output_dir).expanduser()

    if not RAW_DIR.exists():
        raise FileNotFoundError(f"No existe raw_dir: {RAW_DIR}")

    if OUTPUT_DIR.exists():
        if args.overwrite:
            print(f"Limpiando directorio antiguo: {OUTPUT_DIR}")
            shutil.rmtree(OUTPUT_DIR)
        else:
            raise RuntimeError(
                f"El directorio de salida ya existe: {OUTPUT_DIR}\n"
                f"Si quieres regenerarlo, ejecuta con --overwrite."
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_dfs = []

    for batch_dir in sorted(RAW_DIR.glob("batch*")):
        if not batch_dir.is_dir():
            continue

        print(f"Procesando {batch_dir.name}...")

        for json_file in batch_dir.glob("*.json"):
            try:
                df_temp = process_json(json_file, batch_dir.name)
                if not df_temp.empty:
                    all_dfs.append(df_temp)
            except Exception as e:
                print(f"Error en {json_file}: {e}")

    if not all_dfs:
        raise RuntimeError(f"No se ha leído ningún JSON desde {RAW_DIR}")

    raw_df = pd.concat(all_dfs, ignore_index=True)
    raw_df = raw_df.sort_values(by=["batch", "timestamp"], kind="mergesort")

    print(f"Registros crudos totales: {len(raw_df):,}")
    print("Muestra de datos crudos:")
    print(raw_df.head())

    #Deduplicación
    subset_unique = ["timestamp", "sensor_id", "metric"]
    dupes_count = raw_df.duplicated(subset=subset_unique, keep="last").sum()
    print(f"Duplicados detectados (solapamiento de batches): {dupes_count:,}")

    clean_df = raw_df.drop_duplicates(subset=subset_unique, keep="last").copy()
    print(f"Registros finales únicos: {len(clean_df):,}")
    print(f"Se han eliminado {len(raw_df) - len(clean_df)} filas redundantes.")

    before = len(clean_df)
    clean_df = clean_df.dropna(subset=["timestamp", "value"])
    print(f"Filas eliminadas por NaN en timestamp/value: {before - len(clean_df):,}")

    # Categorías de dictionary encoding en Parquet
    for col in ["sensor_id", "metric", "unit", "batch"]:
        clean_df[col] = clean_df[col].astype("category")

    print("Guardando Parquet (dataset particionado por batch)...")

    clean_df.to_parquet(
        OUTPUT_DIR,
        partition_cols=["batch"],
        engine="pyarrow",
        compression="snappy",
        index=False,
    )

    print(f"Dataset guardado en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
