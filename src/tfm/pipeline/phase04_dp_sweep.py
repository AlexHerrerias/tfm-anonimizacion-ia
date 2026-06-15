"""Fase 4 — Barrido DP con diffprivlib: ε ∈ {0.1, 0.5, 1, 5, 10}, veinte
repeticiones por punto (config.DP_SEEDS), sobre LR y Naive Bayes Gaussian.
"""

import pandas as pd

from tfm import config
from tfm.data_loader import load_clean_reduced
from tfm.differential_privacy import sweep_dp
from tfm.fairness import fairness_per_race_dp
from tfm.plotting import plot_dp_degradation, plot_dp_vs_kanon, plot_fairness_curves
from tfm.preprocessing import (
    fit_scaler,
    load_baselines,
    percentile_bounds,
    percentile_data_norm,
    stratified_split,
)


def run() -> None:
    config.ensure_dirs()

    df = load_clean_reduced()
    X_train, X_test, y_train, y_test = stratified_split(df)
    scaler = fit_scaler(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    data_norm = percentile_data_norm(X_train_scaled, percentile=95)
    bounds = percentile_bounds(X_train_scaled)
    print(f"data_norm (p95): {data_norm:.4f}")

    # Sweep agregado
    df_dp = sweep_dp(
        X_train_scaled=X_train_scaled,
        X_test_scaled=X_test_scaled,
        y_train=y_train.values,
        y_test=y_test.values,
        data_norm=data_norm,
        bounds=bounds,
    )
    df_dp.to_csv(config.RESULTS_DP_DIR / "resultados_dp.csv", index=False)

    aggregated = df_dp.groupby(["modelo", "epsilon"]).agg(
        Acc_mean=("Accuracy", "mean"), Acc_std=("Accuracy", "std"),
        F1_mean=("F1-Score", "mean"), F1_std=("F1-Score", "std"),
    ).round(4)
    print("\nResumen mean ± std por (modelo, ε):")
    print(aggregated.to_string())

    plot_dp_degradation(
        df_results_aggregated=aggregated,
        baselines=load_baselines(),
        output_path=config.RESULTS_DIR / "comparativa_dp.png",
    )

    # Comparativa cruzada DP vs k-anonimidad (requiere la fase 3 ya ejecutada)
    kanon_csv = config.RESULTS_KANON_DIR / "resultados_kanon.csv"
    if kanon_csv.exists():
        df_kanon = pd.read_csv(kanon_csv)
        df_kanon_lr = df_kanon[df_kanon["Modelo"] == "Regresión Logística"][["k", "Accuracy"]]
        baseline_acc = float(df_kanon_lr[df_kanon_lr["k"] == 0]["Accuracy"].iloc[0])
        plot_dp_vs_kanon(
            df_kanon_lr.sort_values("k"),
            aggregated.loc["Regresión Logística"],
            baseline_acc,
            config.RESULTS_DIR / "comparativa_dp_vs_kanon.png",
        )
    else:
        print("Aviso: sin resultados_kanon.csv; se omite comparativa_dp_vs_kanon.png")

    # Fairness por raza
    df_test_raw = df.loc[X_test.index].copy()
    df_fairness = fairness_per_race_dp(
        X_train_scaled=X_train_scaled,
        X_test_scaled=X_test_scaled,
        y_train=y_train.values,
        y_test=y_test.values,
        races_test=df_test_raw["race"].values,
        data_norm=data_norm,
    )
    df_fairness.to_csv(config.RESULTS_DP_DIR / "fairness_dp.csv", index=False)
    plot_fairness_curves(
        df_fairness,
        x_column="epsilon",
        x_label=r"$\epsilon$",
        output_path=config.RESULTS_DIR / "fairness_dp.png",
        title=r"Disparidad por subgrupo race en LR-DP (mean$\pm$std, n=20)",
    )


if __name__ == "__main__":
    run()
