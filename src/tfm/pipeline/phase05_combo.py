"""Fase 5 — Defensa en profundidad: sweep DP ε ∈ {0.1, 1, 10} sobre el
conjunto previamente anonimizado con k=10, con fairness por subgrupo race
y figuras comparativas frente a la DP sola.
"""

import pandas as pd

from tfm import config
from tfm.arx_io import load_arx_arrays
from tfm.data_loader import load_clean_reduced
from tfm.differential_privacy import dp_on_kanonimized
from tfm.fairness import fairness_per_race_dp
from tfm.plotting import plot_combo_degradation, plot_fairness_combo
from tfm.preprocessing import (
    binarize_target,
    percentile_data_norm,
    raw_frames_from_split,
    stratified_split,
)

COMBO_EPSILONS = [0.1, 1.0, 10.0]


def run() -> None:
    config.ensure_dirs()

    df = load_clean_reduced()
    X_train, X_test, _, _ = stratified_split(df)
    _, df_test_raw = raw_frames_from_split(df, X_train.index, X_test.index)

    filepath = config.ARX_OUTPUTS_DIR / config.arx_output_filename(10)
    if not filepath.exists():
        raise FileNotFoundError(
            f"No se encuentra {filepath}. Ejecuta antes la fase ARX para k=10."
        )

    df_combo = dp_on_kanonimized(
        filepath_kanon=filepath,
        df_test_raw=df_test_raw,
    )
    df_combo.to_csv(config.RESULTS_DP_DIR / "resultados_combo.csv", index=False)

    aggregated = df_combo.groupby(["modelo", "epsilon"])["Accuracy"].agg(["mean", "std"]).round(4)
    print("Resultados de la combinación k=10 + DP (Accuracy mean ± std):")
    print(aggregated.to_string())

    # Fairness por subgrupo sobre la combinación (misma batería que la fase 4)
    arrays = load_arx_arrays(filepath, df_test_raw)
    data_norm = percentile_data_norm(arrays.X_anon_scaled, percentile=95)
    y_test = binarize_target(df_test_raw[config.TARGET_COLUMN]).values
    df_fairness_combo = fairness_per_race_dp(
        arrays.X_anon_scaled, arrays.X_test_scaled, arrays.y_anon, y_test,
        df_test_raw["race"].values, data_norm, epsilons=COMBO_EPSILONS,
    )
    df_fairness_combo.to_csv(config.RESULTS_DP_DIR / "fairness_combo.csv", index=False)

    # Figuras comparativas (requieren CSVs de las fases 3 y 4)
    dp_csv = config.RESULTS_DP_DIR / "resultados_dp.csv"
    kanon_csv = config.RESULTS_KANON_DIR / "resultados_kanon.csv"
    fairness_dp_csv = config.RESULTS_DP_DIR / "fairness_dp.csv"
    if dp_csv.exists() and kanon_csv.exists():
        df_dp = pd.read_csv(dp_csv)
        agg_lr = (df_dp[df_dp["modelo"] == "Regresión Logística"]
                  .groupby("epsilon")["Accuracy"].agg(Acc_mean="mean", Acc_std="std"))
        combo_lr = (df_combo[df_combo["modelo"] == "Regresión Logística"]
                    .groupby("epsilon")["Accuracy"].agg(Acc_mean="mean", Acc_std="std"))
        df_kanon = pd.read_csv(kanon_csv)
        lr_rows = df_kanon[df_kanon["Modelo"] == "Regresión Logística"]
        baseline_acc = float(lr_rows[lr_rows["k"] == 0]["Accuracy"].iloc[0])
        k10_acc = float(lr_rows[lr_rows["k"] == 10]["Accuracy"].iloc[0])
        plot_combo_degradation(
            agg_lr, combo_lr, baseline_acc, k10_acc,
            config.RESULTS_DIR / "comparativa_combo.png",
        )
        if fairness_dp_csv.exists():
            plot_fairness_combo(
                pd.read_csv(fairness_dp_csv), df_fairness_combo, baseline_acc,
                config.RESULTS_DIR / "fairness_combo.png",
            )
    else:
        print("Aviso: faltan CSVs de las fases 3/4; se omiten las figuras comparativas")


if __name__ == "__main__":
    run()
