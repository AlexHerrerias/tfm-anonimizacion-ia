"""Fase 3 — Evaluación post-ARX y análisis de equidad.

Lee los CSVs anonimizados arx_output_k<k>.csv producidos manualmente
en ARX Desktop, reentrena los modelos sobre cada versión y produce
las tablas y figuras del Capítulo 4.
"""

import pandas as pd

from tfm import config
from tfm.data_loader import load_clean_reduced
from tfm.fairness import fairness_per_race_kanon, loss_disparity_table
from tfm.kanon import evaluate_kanon
from tfm.plotting import plot_fairness_curves, plot_kanon_degradation
from tfm.preprocessing import raw_frames_from_split, stratified_split


def run() -> None:
    config.ensure_dirs()

    df = load_clean_reduced()
    X_train, X_test, _, _ = stratified_split(df)
    df_train_raw, df_test_raw = raw_frames_from_split(df, X_train.index, X_test.index)

    # Tabla de utilidad por modelo y nivel de k
    df_baseline = pd.read_csv(config.RESULTS_KANON_DIR / "baseline.csv")
    df_baseline = df_baseline.assign(pct_suprimido=0,
                                       filas_efectivas=len(X_train),
                                       columnas_OHE=X_train.shape[1])
    results = [df_baseline]

    for k in config.K_VALUES:
        filepath = config.ARX_OUTPUTS_DIR / config.arx_output_filename(k)
        if filepath.exists():
            results.append(evaluate_kanon(filepath, k, df_test_raw))
        else:
            print(f"Aviso: no se encuentra {filepath}, se omite k={k}")

    df_full = pd.concat(results, ignore_index=True)
    df_full.to_csv(config.RESULTS_KANON_DIR / "resultados_kanon.csv", index=False)
    plot_kanon_degradation(df_full, config.RESULTS_DIR / "comparativa_kanon.png")
    print("Tabla de utilidad guardada en resultados_kanon.csv")

    # Análisis de fairness por raza
    fairness_partes = [fairness_per_race_kanon(None, 0, df_train_raw, df_test_raw)]
    for k in config.K_VALUES:
        filepath = config.ARX_OUTPUTS_DIR / config.arx_output_filename(k)
        if filepath.exists():
            fairness_partes.append(fairness_per_race_kanon(filepath, k, df_train_raw, df_test_raw))

    df_fairness = pd.concat(fairness_partes, ignore_index=True)
    df_fairness.to_csv(config.RESULTS_KANON_DIR / "fairness_kanon.csv", index=False)
    plot_fairness_curves(
        df_fairness,
        x_column="k",
        x_label="k",
        output_path=config.RESULTS_DIR / "fairness_kanon.png",
        title="Disparidad por subgrupo race a lo largo del barrido de k",
    )

    # Loss Disparity por k (los valores se citan en la memoria)
    df_disparity = loss_disparity_table(df_fairness, x_column="k")
    df_disparity.to_csv(config.RESULTS_KANON_DIR / "loss_disparity_kanon.csv", index=False)
    print("\nLoss Disparity por k:")
    for _, row in df_disparity.iterrows():
        print(f"  k={int(row['k'])}: {row['loss_disparity']:.4f}")


if __name__ == "__main__":
    run()
