"""Fase 6 — Auditoría adversarial mediante MIA Black-Box.

Audita seis escenarios sobre Regresión Logística: baseline, k=10, k=50,
y DP con ε ∈ {0.1, 1, 10}.
"""

from typing import List

import pandas as pd

from tfm import config
from tfm.arx_io import load_arx_arrays
from tfm.data_loader import load_clean_reduced
from tfm.mia import run_mia_blackbox
from tfm.models import fit_lr_baseline, fit_lr_dp
from tfm.plotting import plot_mia_bars
from tfm.preprocessing import (
    fit_scaler,
    percentile_data_norm,
    raw_frames_from_split,
    stratified_split,
)


def run() -> None:
    config.ensure_dirs()

    df = load_clean_reduced()
    X_train, X_test, y_train, y_test = stratified_split(df)
    scaler = fit_scaler(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    _, df_test_raw = raw_frames_from_split(df, X_train.index, X_test.index)

    data_norm = percentile_data_norm(X_train_scaled, percentile=95)

    results: List[dict] = []

    # Baseline LR
    lr_baseline = fit_lr_baseline(X_train_scaled, y_train.values)
    results.append({
        "escenario": "Baseline LR",
        **run_mia_blackbox(lr_baseline, X_train_scaled, y_train.values, X_test_scaled, y_test.values),
    })

    # k=10 y k=50
    for k_value in (10, 50):
        filepath = config.ARX_OUTPUTS_DIR / config.arx_output_filename(k_value)
        if not filepath.exists():
            print(f"Aviso: no se encuentra {filepath}, se omite k={k_value}")
            continue
        arrays = load_arx_arrays(filepath, df_test_raw)
        lr_k = fit_lr_baseline(arrays.X_anon_scaled, arrays.y_anon)
        results.append({
            "escenario": f"LR · k={k_value}",
            **run_mia_blackbox(
                lr_k, arrays.X_anon_scaled, arrays.y_anon, arrays.X_test_scaled, y_test.values
            ),
        })

    # DP ε=0.1, ε=1 y ε=10
    for epsilon in (0.1, 1.0, 10.0):
        lr_dp = fit_lr_dp(X_train_scaled, y_train.values, epsilon, data_norm)
        results.append({
            "escenario": f"LR · DP ε={epsilon}",
            **run_mia_blackbox(lr_dp, X_train_scaled, y_train.values, X_test_scaled, y_test.values),
        })

    df_mia = pd.DataFrame(results)
    df_mia.to_csv(config.RESULTS_MIA_DIR / "mia_results.csv", index=False)
    plot_mia_bars(df_mia, config.RESULTS_DIR / "mia_results.png")
    print(df_mia.round(4).to_string(index=False))


if __name__ == "__main__":
    run()
