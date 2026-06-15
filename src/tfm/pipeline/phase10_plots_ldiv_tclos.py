"""Fase 10 — Figuras de la extensión l-diversidad / t-closeness a partir de los
CSVs de la fase 8: comparativa_ldiv, comparativa_tclos, triples_vs_singles y
fairness_ldiv_tclos (todas .png).
"""

import pandas as pd

from tfm import config
from tfm.plotting import (
    plot_fairness_ldiv_tclos,
    plot_ldiv_degradation,
    plot_tclos_degradation,
    plot_triples_vs_singles,
)


def run() -> None:
    config.ensure_dirs()

    results_csv = config.RESULTS_LDIV_TCLOS_DIR / "resultados_ldiv_tclos.csv"
    fairness_csv = config.RESULTS_LDIV_TCLOS_DIR / "fairness_ldiv_tclos.csv"
    if not results_csv.exists() or not fairness_csv.exists():
        raise FileNotFoundError(
            "Ejecuta primero `tfm run 08` para producir los CSVs."
        )

    df_results = pd.read_csv(results_csv)
    df_fairness = pd.read_csv(fairness_csv)
    baseline_k5 = df_results[df_results["tipo"] == "k_solo"][
        ["Modelo", "Accuracy", "F1-Score"]
    ].copy()

    plot_ldiv_degradation(
        df_results, baseline_k5,
        config.RESULTS_DIR / "comparativa_ldiv.png",
    )
    plot_tclos_degradation(
        df_results, baseline_k5,
        config.RESULTS_DIR / "comparativa_tclos.png",
    )
    plot_triples_vs_singles(
        df_results, baseline_k5,
        config.RESULTS_DIR / "triples_vs_singles.png",
    )
    plot_fairness_ldiv_tclos(
        df_fairness,
        config.RESULTS_DIR / "fairness_ldiv_tclos.png",
    )

    print("Figuras generadas en", config.RESULTS_DIR)


if __name__ == "__main__":
    run()
