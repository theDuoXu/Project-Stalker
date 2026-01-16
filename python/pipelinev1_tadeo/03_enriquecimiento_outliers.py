import argparse
from pathlib import Path
import shutil
import cudf
import cupy as cp
import numpy as np

def parse_args():
    p = argparse.ArgumentParser(
        description="Enriquecimiento con imputación, rolling z-score e IsolationForest"
    )
    p.add_argument(
        "--input_dir",
        default="~/Projects/proyecto-computacion-I/data/datasets/processed/02_qc/measurements",
        help="Directorio Parquet de entrada (particionado)",
    )
    p.add_argument(
        "--output_dir",
        default="~/Projects/proyecto-computacion-I/data/datasets/processed/03_enriched/measurements",
        help="Directorio Parquet de salida (particionado por batch)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Si se indica, borra el output_dir antes de regenerar el dataset",
    )
    p.add_argument(
        "--with_iforest",
        action="store_true",
        help="Ejecuta IsolationForest-",
    )
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
                f"Si se quiere regenerar, ejecuta con --overwrite."
            )

    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)

    try:
        import rmm
        rmm.reinitialize(managed_memory=True)
    except Exception:
        pass

    # Configuración de imputación
    GROUP_A_LINEAR_6H = {"TEMPERATURA", "NIVEL", "CONDUCTIVIDAD"}
    GROUP_B_CUBIC_3H  = {"OXIGENO DISUELTO", "CLOROFILA", "FICOCIANINAS", "PH"}
    GROUP_C_LINEAR_2H = {"TURBIDEZ", "AMONIO", "NITRATOS", "FOSFATOS", "CARBONO ORGANICO"}

    # configuración de z-score
    metrics_log = [
        "AMONIO", "TURBIDEZ", "NITRATOS", "FOSFATOS",
        "CLOROFILA", "FICOCIANINAS", "CONDUCTIVIDAD", "CARBONO ORGANICO"
    ]
    WINDOW_SIZE = 96
    MIN_PERIODS = 24
    MAX_ZSCORE = 6
    EPSILON = 1e-6


    def _nan_runs(isna: cp.ndarray):
        """Encuentras los tramos consecutvos de NaN y devuelve las
        posiciones de inicio y fin de cada hueco"""
        if isna.size == 0:
            return [], []
        left = cp.concatenate([cp.array([True]), ~isna[:-1]])
        right = cp.concatenate([~isna[1:], cp.array([True])])
        starts = cp.where(isna & left)[0]
        ends = cp.where(isna & right)[0]
        return starts.get().tolist(), ends.get().tolist()


    def _fill_linear(v: cp.ndarray, s: int, e: int):
        """Rellena el hueco v[s:e+1] lineal dibujando una recta 
        entre v[s-1] y v[e+1]."""
        p1 = v[s - 1]
        p2 = v[e + 1]
        L = e - s + 1
        filled = cp.linspace(p1, p2, L + 2)[1:-1]
        v[s:e + 1] = filled


    def _fill_cubic_catmull_rom(v: cp.ndarray, s: int, e: int):
        """Rellena el hueco con una curva cúbica suave (Catmull-Rom) alrededor del hueco."""
        L = e - s + 1
        p1 = v[s - 1]
        p2 = v[e + 1]
        p0 = v[s - 2] if (s - 2) >= 0 else p1
        p3 = v[e + 2] if (e + 2) < v.size else p2

        if bool(cp.isnan(p0)) or bool(cp.isnan(p1)) or bool(cp.isnan(p2)) or bool(cp.isnan(p3)):
            _fill_linear(v, s, e)
            return

        t = cp.arange(1, L + 1, dtype=cp.float32) / (L + 1)
        t2 = t * t
        t3 = t2 * t

        filled = 0.5 * (
            2 * p1
            + (-p0 + p2) * t
            + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
            + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
        )
        v[s:e + 1] = filled


    def impute_values_gpu(df_gpu: cudf.DataFrame) -> cudf.DataFrame:
        """
        Imputa NaNs en df_gpu['value'] según la política A/B/C del proyecto.
        Se supone que el dataset ordenado por sensor_id, metric, timestamp con index 0..n-1.
        """
        n = len(df_gpu)
        value = df_gpu["value"].astype("float64").to_cupy()

        was_imputed = cp.zeros(n, dtype=cp.bool_)
        method_code = cp.zeros(n, dtype=cp.int8) 

        # Clave de grupo contigua
        g = (df_gpu["sensor_id"].astype("str") + "|" + df_gpu["metric"].astype("str"))
        g_prev = g.shift(1)

        # Nuevo grupo cuando cambia o cuando el previo es null
        change = (g != g_prev) | g_prev.isna()

        starts = cp.where(change.to_cupy())[0]
        ends = cp.concatenate([starts[1:] - 1, cp.array([n - 1], dtype=starts.dtype)])

        for i in range(int(starts.size)):
            a = int(starts[i].item())
            b = int(ends[i].item())

            key = str(g.iloc[a])
            metric = key.split("|", 1)[1]

            if metric in GROUP_A_LINEAR_6H:
                limit = 6
                mode = "linear"
            elif metric in GROUP_B_CUBIC_3H:
                limit = 3
                mode = "cubic"
            elif metric in GROUP_C_LINEAR_2H:
                limit = 2
                mode = "linear"
            else:
                continue

            v = value[a:b + 1]
            if v.size < 3:
                continue

            isna = cp.isnan(v)
            if not bool(isna.any()):
                continue

            run_starts, run_ends = _nan_runs(isna)

            for rs, re in zip(run_starts, run_ends):
                L = re - rs + 1
                if L > limit:
                    continue
                if rs == 0 or re == (v.size - 1):
                    continue
                if bool(cp.isnan(v[rs - 1])) or bool(cp.isnan(v[re + 1])):
                    continue

                if mode == "linear":
                    _fill_linear(v, rs, re)
                    was_imputed[a + rs:a + re + 1] = True
                    method_code[a + rs:a + re + 1] = 1
                else:
                    _fill_cubic_catmull_rom(v, rs, re)
                    was_imputed[a + rs:a + re + 1] = True
                    method_code[a + rs:a + re + 1] = 2

            value[a:b + 1] = v

        out = df_gpu.copy()
        out["value_imputed"] = cudf.Series(value)
        out["was_imputed"] = cudf.Series(was_imputed)
        out["impute_method_code"] = cudf.Series(method_code)
        return out


    df_gpu = cudf.read_parquet(INPUT_DIR)
    print(f"Filas en dataset de entrada: {len(df_gpu):,}")

    df_gpu["timestamp"] = cudf.to_datetime(df_gpu["timestamp"], errors="coerce")
    df_gpu = df_gpu.dropna(subset=["timestamp"])
    df_gpu = df_gpu.sort_values(["sensor_id", "metric", "timestamp"]).reset_index(drop=True)

    print("Imputando en GPU...")
    df_gpu = impute_values_gpu(df_gpu)
    print(f"Total imputadas: {int(df_gpu['was_imputed'].sum()):,}")

    # Transformación de logs
    df_gpu["trans_value"] = df_gpu["value"]              
    df_gpu["trans_value_if"] = df_gpu["value_imputed"]  

    mask_log_v = df_gpu["metric"].isin(metrics_log) & df_gpu["value"].notna()
    df_gpu.loc[mask_log_v, "trans_value"] = cp.log1p(df_gpu.loc[mask_log_v, "value"])

    mask_log_i = df_gpu["metric"].isin(metrics_log) & df_gpu["value_imputed"].notna()
    df_gpu.loc[mask_log_i, "trans_value_if"] = cp.log1p(df_gpu.loc[mask_log_i, "value_imputed"])


    # Rolling z-score GPU
    grouped = df_gpu.groupby(["sensor_id", "metric"])
    roll_mean = grouped["trans_value"].rolling(WINDOW_SIZE, center=True, min_periods=MIN_PERIODS).mean()
    roll_std  = grouped["trans_value"].rolling(WINDOW_SIZE, center=True, min_periods=MIN_PERIODS).std()

    roll_mean_values = roll_mean.reset_index(level=[0, 1], drop=True).sort_index()
    roll_std_values  = roll_std.reset_index(level=[0, 1], drop=True).sort_index()
    df_gpu = df_gpu.sort_index()

    numerator = df_gpu["trans_value"] - roll_mean_values
    denominator = roll_std_values.where(roll_std_values > EPSILON, EPSILON)
    z_scores = (numerator / denominator).fillna(0)


    df_gpu["z_score"] = z_scores
    df_gpu["z_score_abs"] = z_scores.abs()
    df_gpu["is_outlier_zscore"] = df_gpu["z_score_abs"] > MAX_ZSCORE

    print(f"Outliers (|z|>{MAX_ZSCORE}): {int(df_gpu['is_outlier_zscore'].sum()):,}")
    print(f"Z max: {float(df_gpu['z_score'].max()):.4f} | Z min: {float(df_gpu['z_score'].min()):.4f}")

    #IsolationForest CPU
    if args.with_iforest:
        import pandas as pd
        from sklearn.ensemble import IsolationForest

        print("\n[IFOREST] Pasando a CPU para pivot + modelos...")
        df_pd = df_gpu.to_pandas()

        df_pivot = df_pd.pivot_table(
            index=["sensor_id", "timestamp"],
            columns="metric",
            values="trans_value_if",
        )
        print(f"[IFOREST] Pivot shape: {df_pivot.shape}")

        sensor_profiles = df_pivot.groupby("sensor_id").apply(
            lambda x: tuple(sorted(x.dropna(axis=1, how="all").columns))
        )
        unique_profiles = sensor_profiles.unique()
        print(f"[IFOREST] Perfiles distintos: {len(unique_profiles)}")

        results = []
        for i, profile in enumerate(unique_profiles):
            sensors_in_profile = sensor_profiles[sensor_profiles == profile].index
            print(f"\n--- Grupo {i+1}/{len(unique_profiles)}: {profile} ---")

            subset = df_pivot.loc[
                df_pivot.index.get_level_values("sensor_id").isin(sensors_in_profile),
                list(profile),
            ].copy()

            subset = subset.ffill().dropna()
            if len(subset) < 100:
                print("Muy pocos datos, saltando.")
                continue

            iso = IsolationForest(
                n_estimators=100,
                contamination=0.005,
                n_jobs=-1,
                random_state=69420,
            )
            iso.fit(subset)

            preds = iso.predict(subset)
            scores = iso.decision_function(subset)

            scored = pd.DataFrame(index=subset.index)
            scored["is_anomaly_multivariate"] = (preds == -1)
            scored["normality_score"] = scores
            results.append(scored)

            print(f"   Detectadas {int(scored['is_anomaly_multivariate'].sum()):,} anomalías.")

        if results:
            df_scores = pd.concat(results).reset_index()
            df_pd = df_pd.merge(df_scores, on=["sensor_id", "timestamp"], how="left")
            df_pd["is_anomaly_multivariate"] = df_pd["is_anomaly_multivariate"].fillna(False)
        else:
            df_pd["is_anomaly_multivariate"] = False
            df_pd["normality_score"] = np.nan

        df_gpu = cudf.from_pandas(df_pd)

    # Guardado final
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nGuardando dataset enriquecido en {OUTPUT_DIR} (partition_cols=batch)...")

    df_out = df_gpu.to_pandas()
    df_out.to_parquet(
        OUTPUT_DIR,
        partition_cols=["batch"],
        engine="pyarrow",
        compression="snappy",
        index=False,
    )

    print("Dataset enriquecido generado OK.")

if __name__ == "__main__":
    main()