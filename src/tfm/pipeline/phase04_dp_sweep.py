"""Fase 4 — Barrido de Privacidad Diferencial con diffprivlib.

Ejecuta ε ∈ {0.1, 0.5, 1, 5, 10} con veinte repeticiones por punto
(semillas dispersas, ver config.DP_SEEDS) sobre los modelos de
Regresión Logística y Naive Bayes Gaussian.
"""

from tfm import config
from tfm.data_loader import load_clean_reduced
from tfm.differential_privacy import sweep_dp
from tfm.fairness import fairness_per_race_dp
from tfm.plotting import plot_dp_degradation, plot_fairness_curves
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
