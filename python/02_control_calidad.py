import argparse
from pathlib import Path
import shutil
import cudf
import cupy as cp


def parse_args():
    p = argparse.ArgumentParser(
        description="QC en GPU (cuDF): límites físicos + coherencia temporal"
    )
    p.add_argument(
        "--input_dir",
        default="~/Projects/proyecto-computacion-I/data/datasets/processed/01_ingested/measurements",
    )
    p.add_argument(
        "--output_dir",
        default="~/Projects/proyecto-computacion-I/data/datasets/processed/02_qc/measurements",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()

    INPUT_DIR = Path(args.input_dir).expanduser()
    OUTPUT_DIR = Path(args.output_dir).expanduser()

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"No existe input_dir: {INPUT_DIR}")

    if OUTPUT_DIR.exists():
        if args.overwrite:
            print(f"Limpiando directorio antiguo: {OUTPUT_DIR}")
            shutil.rmtree(OUTPUT_DIR)
        else:
            raise RuntimeError(
                f"El directorio de salida ya existe: {OUTPUT_DIR}\n"
                f"Si quieres regenerarlo, ejecuta con --overwrite."
            )

    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)

    try:
        import rmm
        rmm.reinitialize(managed_memory=True)
    except Exception:
        pass

    HARD_LIMITS = {
        "AMONIO": (0.0, 20.0),
        "CARBONO ORGANICO": (0.0, 50.0),
        "CLOROFILA": (0.0, 500.0),
        "CONDUCTIVIDAD": (20.0, 5000.0),
        "FICOCIANINAS": (0.0, 1000.0),
        "FOSFATOS": (0.0, 10.0),
        "NITRATOS": (0.0, 250.0),
        "NIVEL": (0.0, 20.0),
        "OXIGENO DISUELTO": (0.0, 20.0),
        "PH": (4.0, 10.5),
        "TEMPERATURA": (0.0, 38.0),
        "TURBIDEZ": (0.0, 2000.0),
    }

    NO_EXACT_ZERO = {"CARBONO ORGANICO", "PH", "TEMPERATURA"}
    MAX_TEMP_JUMP_PER_HOUR = 10.0

    df = cudf.read_parquet(INPUT_DIR)
    print(f"Filas en dataset de entrada: {len(df):,}")

    required_cols = {"timestamp", "sensor_id", "metric", "value", "batch"}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"Faltan columnas necesarias en entrada: {sorted(missing)}")

    subset_unique = ["timestamp", "sensor_id", "metric"]
    dupes = df.duplicated(subset=subset_unique, keep="last").sum()
    print(f"Duplicados encontrados en entrada (esperado 0): {int(dupes):,}")

    # Tipos
    df["timestamp"] = cudf.to_datetime(df["timestamp"], errors="coerce")
    df["value"] = df["value"].astype("float64")

    # Se quitan filas con timestamp inválido (NaT)
    before = len(df)
    df = df.dropna(subset=["timestamp"])
    if len(df) != before:
        print(f"Filas eliminadas por timestamp inválido (NaT): {before - len(df):,}")

    # Flags
    df["qc_physical_invalid"] = False
    df["qc_temporal_invalid"] = False

    # Límites físicos
    for metric, (mn, mx) in HARD_LIMITS.items():
        m = df["metric"] == metric
        bad = m & ((df["value"] < mn) | (df["value"] > mx))
        if bool(bad.any()):
            df.loc[bad, "qc_physical_invalid"] = True
            df.loc[bad, "value"] = cp.nan

    # Aqui no es cero exacto
    bad0 = df["metric"].isin(list(NO_EXACT_ZERO)) & (df["value"] == 0)
    if bool(bad0.any()):
        df.loc[bad0, "qc_physical_invalid"] = True
        df.loc[bad0, "value"] = cp.nan

    #Coherencia temporal en temperatura
    temp_mask = df["metric"] == "TEMPERATURA"
    temp = df.loc[temp_mask, ["sensor_id", "timestamp", "value"]].copy()
    temp = temp.sort_values(["sensor_id", "timestamp"])

    temp["prev"] = temp.groupby("sensor_id")["value"].shift(1)
    temp["delta"] = (temp["value"] - temp["prev"]).abs()

    bad_temp = (temp["delta"] > MAX_TEMP_JUMP_PER_HOUR) & temp["prev"].notna()
    if bool(bad_temp.any()):
        bad_idx = temp.index[bad_temp]
        df.loc[bad_idx, "qc_temporal_invalid"] = True
        df.loc[bad_idx, "value"] = cp.nan

    phys = int(df["qc_physical_invalid"].sum())
    temp_bad = int(df["qc_temporal_invalid"].sum())
    n_nans = int(df["value"].isna().sum())

    print(f"Invalidaciones físicas: {phys:,}")
    print(f"Invalidaciones temporales (temperatura): {temp_bad:,}")
    print(f"Total NaNs en value tras QC: {n_nans:,}")

    df = df.reset_index(drop=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Guardando dataset QC en: {OUTPUT_DIR} (partition_cols=batch)")
    df_out = df.to_pandas()
    df_out.to_parquet(
        OUTPUT_DIR,
        engine="pyarrow",
        compression="snappy",
        index=False,
        partition_cols=["batch"],
    )

    print("QC GPU generado OK.")


if __name__ == "__main__":
    main()
