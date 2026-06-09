"""Tests de significancia para las afirmaciones del Capítulo 4."""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2 as chi2_distribution
from scipy.stats import wilcoxon


def mcnemar_test(y_true: np.ndarray, predictions_a: np.ndarray, predictions_b: np.ndarray) -> Tuple[float, int, int, float]:
    """Test de McNemar con corrección de continuidad de Edwards.

    Devuelve (p_value, b, c, chi_squared) donde b cuenta los casos en
    los que el primer clasificador acierta y el segundo falla, y c
    los inversos.
    """
    correct_a = (predictions_a == y_true)
    correct_b = (predictions_b == y_true)
    b = int(((correct_a) & (~correct_b)).sum())
    c = int(((~correct_a) & (correct_b)).sum())
    if b + c == 0:
        return 1.0, 0, 0, 0.0
    chi_squared = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = float(1 - chi2_distribution.cdf(chi_squared, df=1))
    return p_value, b, c, float(chi_squared)


def wilcoxon_one_sample(values: np.ndarray, reference: float) -> Tuple[float, float]:
    """Test de Wilcoxon de una muestra frente a una constante de referencia.

    Devuelve (statistic, p_value). Con n=20 (configuración del TFM) el menor
    p alcanzable a dos colas ronda 1,9 x 10^-6.
    """
    diffs = np.asarray(values) - reference
    try:
        statistic, p_value = wilcoxon(diffs)
        return float(statistic), float(p_value)
    except ValueError:
        return 0.0, 1.0


def apply_bonferroni(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Añade columnas de corrección Bonferroni a una tabla de tests.

    Devuelve una copia del DataFrame con dos columnas adicionales:
      * `p_bonferroni`: p-valor ajustado (= p_value * N, capado a 1.0)
      * `significativo_bonf`: True si p_bonferroni < alpha

    N es el número de tests en la tabla. Esta corrección es conservadora
    pero fácil de interpretar; alternativamente puede usarse Benjamini-Hochberg
    para control de FDR cuando el número de tests es elevado.
    """
    df = df.copy()
    n_tests = len(df)
    df["p_bonferroni"] = (df["p_value"] * n_tests).clip(upper=1.0)
    df["significativo_bonf"] = df["p_bonferroni"] < alpha
    return df


def build_mcnemar_table(
    baseline_predictions: Dict[str, np.ndarray],
    anonymized_predictions: Dict[Tuple[str, int], np.ndarray],
    y_test: np.ndarray,
) -> pd.DataFrame:
    """Construye la tabla completa de tests McNemar baseline vs (modelo, k).

    Incluye la corrección Bonferroni sobre la familia completa de tests
    (N = nº de filas) para controlar el error tipo I familywise.
    """
    rows: List[Dict] = []
    for (model_name, k_value), predictions in anonymized_predictions.items():
        p_value, b, c, chi_squared = mcnemar_test(
            y_test,
            baseline_predictions[model_name],
            predictions,
        )
        rows.append({
            "modelo": model_name,
            "k": k_value,
            "b": b,
            "c": c,
            "chi2": round(chi_squared, 4),
            "p_value": p_value,
            "significativo_005": p_value < 0.05,
        })
    return apply_bonferroni(pd.DataFrame(rows))


def build_wilcoxon_table(
    df_dp_results: pd.DataFrame,
    baselines: Dict[str, float],
) -> pd.DataFrame:
    """Tabla de Wilcoxon DP frente a baselines globales con Bonferroni."""
    rows: List[Dict] = []
    for model_name, baseline_acc in baselines.items():
        subset_model = df_dp_results[df_dp_results["modelo"] == model_name]
        for epsilon in sorted(subset_model["epsilon"].unique()):
            sample = subset_model[subset_model["epsilon"] == epsilon]["Accuracy"].values
            _, p_value = wilcoxon_one_sample(sample, baseline_acc)
            rows.append({
                "modelo": model_name,
                "epsilon": epsilon,
                "mean_acc": float(np.mean(sample)),
                "std_acc": float(np.std(sample)),
                "baseline": baseline_acc,
                "p_value": p_value,
                "delta_pct": (float(np.mean(sample)) - baseline_acc) / baseline_acc * 100,
                "significativo_005": p_value < 0.05,
            })
    return apply_bonferroni(pd.DataFrame(rows))
