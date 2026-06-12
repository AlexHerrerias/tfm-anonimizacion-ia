"""Fase 12 — Barrido de Privacidad Diferencial LOCAL (LDP).

Perturba el conjunto de entrenamiento registro a registro (k-RR para
categóricas, Laplace acotado para recuentos) con presupuesto user-level
ε ∈ config.LDP_EPSILON_VALUES y veinte réplicas por punto (semillas
dispersas, ver config.DP_SEEDS), entrena Regresión Logística y Naive
Bayes sobre los datos perturbados y evalúa contra el test en claro.

Salidas:
  arx_kit/results/ldp/resultados_ldp.csv   utilidad por (ε, rep, modelo)
  arx_kit/results/ldp/fairness_ldp.csv     Accuracy por subgrupo race (LR)
  arx_kit/results/ldp/wilcoxon_ldp.csv     tests vs baseline con Bonferroni
  results/comparativa_ldp.png              curva de degradación LDP
  results/comparativa_ldp_vs_dp.png        LDP vs DP global (LR)
  results/fairness_ldp.png                 curvas por subgrupo race
"""

import pandas as pd

from tfm import config
from tfm.data_loader import load_clean_reduced
from tfm.local_dp import sweep_ldp
from tfm.plotting import plot_fairness_curves, plot_ldp_degradation, plot_ldp_vs_dp
from tfm.preprocessing import load_baselines, raw_frames_from_split, stratified_split
from tfm.statistical_tests import build_wilcoxon_table


def _aggregate(df_results: pd.DataFrame) -> pd.DataFrame:
    return df_results.groupby(["modelo", "epsilon"]).agg(
        Acc_mean=("Accuracy", "mean"), Acc_std=("Accuracy", "std"),
        F1_mean=("F1-Score", "mean"), F1_std=("F1-Score", "std"),
    ).round(4)


def run() -> None:
    config.ensure_dirs()

    df = load_clean_reduced()
    X_train, X_test, _, y_test = stratified_split(df)
    df_train_raw, df_test_raw = raw_frames_from_split(
        df, X_train.index, X_test.index, coerce_qids=False
    )

    df_ldp, df_fairness = sweep_ldp(df_train_raw, df_test_raw)
    df_ldp.to_csv(config.RESULTS_LDP_DIR / "resultados_ldp.csv", index=False)
    df_fairness.to_csv(config.RESULTS_LDP_DIR / "fairness_ldp.csv", index=False)

    aggregated = _aggregate(df_ldp)
    print("\nResumen mean ± std por (modelo, ε):")
    print(aggregated.to_string())

    # Wilcoxon vs baseline (familia: 2 modelos × |ε|, Bonferroni interno)
    baselines = load_baselines()
    df_wilcoxon = build_wilcoxon_table(
        df_ldp, {name: acc for name, (acc, _) in baselines.items() if name in df_ldp["modelo"].unique()}
    )
    df_wilcoxon.to_csv(config.RESULTS_LDP_DIR / "wilcoxon_ldp.csv", index=False)
    print("\nWilcoxon vs baseline (Bonferroni):")
    print(df_wilcoxon.round(4).to_string(index=False))

    # Figuras
    plot_ldp_degradation(
        df_results_aggregated=aggregated,
        baselines=baselines,
        output_path=config.RESULTS_DIR / "comparativa_ldp.png",
    )

    dp_results_path = config.RESULTS_DP_DIR / "resultados_dp.csv"
    if dp_results_path.exists():
        df_dp = pd.read_csv(dp_results_path)
        majority_acc = float((y_test == 0).mean())
        plot_ldp_vs_dp(
            df_ldp_aggregated=aggregated,
            df_dp_aggregated=_aggregate(df_dp),
            baseline_lr_acc=baselines["Regresión Logística"][0],
            output_path=config.RESULTS_DIR / "comparativa_ldp_vs_dp.png",
            majority_class_acc=majority_acc,
        )
    else:
        print(f"Aviso: no existe {dp_results_path}; ejecuta `tfm run 04` para la comparativa LDP vs DP.")

    plot_fairness_curves(
        df_fairness,
        x_column="epsilon",
        x_label=r"$\epsilon$ (user-level)",
        output_path=config.RESULTS_DIR / "fairness_ldp.png",
        title=rf"Disparidad por subgrupo race en LR-LDP (mean$\pm$std, n={config.N_REPETITIONS_DP})",
    )


if __name__ == "__main__":
    run()
