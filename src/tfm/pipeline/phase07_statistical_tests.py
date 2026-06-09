"""Fase 7 — Tests de significancia estadística.

Calcula los tests de McNemar para los modelos k-anonimizados y los
tests de Wilcoxon de una muestra para los modelos diferencialmente
privados.
"""

import pandas as pd

from tfm import config
from tfm.arx_io import load_arx_arrays
from tfm.data_loader import load_clean_reduced
from tfm.models import build_baseline_models, predict_baselines
from tfm.preprocessing import (
    fit_scaler,
    load_baselines,
    raw_frames_from_split,
    stratified_split,
)
from tfm.statistical_tests import build_mcnemar_table, build_wilcoxon_table


def run() -> None:
    config.ensure_dirs()

    df = load_clean_reduced()
    X_train, X_test, y_train, y_test = stratified_split(df)
    scaler = fit_scaler(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    _, df_test_raw = raw_frames_from_split(df, X_train.index, X_test.index)

    # Predicciones del baseline para cada modelo
    baseline_predictions = predict_baselines(X_train_scaled, X_test_scaled, y_train)

    # Predicciones por modelo y nivel de k
    anonymized_predictions = {}
    for k_value in config.K_VALUES:
        filepath = config.ARX_OUTPUTS_DIR / config.arx_output_filename(k_value)
        if not filepath.exists():
            continue
        arrays = load_arx_arrays(filepath, df_test_raw)
        for name, model in build_baseline_models().items():
            model.fit(arrays.X_anon_scaled, arrays.y_anon)
            anonymized_predictions[(name, k_value)] = model.predict(arrays.X_test_scaled)

    df_mcnemar = build_mcnemar_table(
        baseline_predictions=baseline_predictions,
        anonymized_predictions=anonymized_predictions,
        y_test=y_test.values,
    )
    df_mcnemar.to_csv(config.RESULTS_KANON_DIR / "mcnemar_kanon.csv", index=False)
    print("McNemar (baseline vs k-anon):")
    print(df_mcnemar.round(4).to_string(index=False))

    # Test de Wilcoxon sobre los resultados DP previamente guardados
    dp_path = config.RESULTS_DP_DIR / "resultados_dp.csv"
    if dp_path.exists():
        df_dp = pd.read_csv(dp_path)
        # Baselines leídos de resultados_kanon.csv (k=0) — fuente única de verdad
        baselines_full = load_baselines()
        baselines = {model: acc for model, (acc, _) in baselines_full.items()}
        df_wilcoxon = build_wilcoxon_table(df_dp, baselines)
        df_wilcoxon.to_csv(config.RESULTS_DP_DIR / "wilcoxon_dp.csv", index=False)
        print("\nWilcoxon (DP vs baseline):")
        print(df_wilcoxon.round(4).to_string(index=False))
    else:
        print(f"\nAviso: no se encuentra {dp_path}, se omiten los tests Wilcoxon")


if __name__ == "__main__":
    run()
