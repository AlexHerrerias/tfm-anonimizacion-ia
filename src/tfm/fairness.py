"""Análisis de equidad algorítmica desagregado por subgrupo `race`."""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

import diffprivlib.models as dpm

from tfm import config
from tfm.arx_io import read_arx_csv
from tfm.preprocessing import binarize_target, joint_ohe


def loss_disparity(accuracies_by_subgroup: pd.Series) -> float:
    """Diferencia entre el subgrupo más favorecido y el menos favorecido."""
    return float(accuracies_by_subgroup.max() - accuracies_by_subgroup.min())


def loss_disparity_table(df_fairness: pd.DataFrame, x_column: str = "k") -> pd.DataFrame:
    """Tabla de Loss Disparity (max-min de Accuracy por subgrupo) para cada valor de `x_column`."""
    pivot = df_fairness.pivot_table(index="race", columns=x_column, values="Accuracy", aggfunc="mean")
    rows = [
        {x_column: level, "loss_disparity": loss_disparity(pivot[level])}
        for level in pivot.columns
    ]
    return pd.DataFrame(rows)


def fairness_per_race_kanon(
    filepath: Path,
    k_value: int,
    df_train_raw: pd.DataFrame,
    df_test_raw: pd.DataFrame,
) -> pd.DataFrame:
    """Desagrega la Accuracy de la Regresión Logística por subgrupo `race`."""
    if k_value == 0:
        df_anon = df_train_raw.copy()
    else:
        df_anon, _ = read_arx_csv(filepath)

    y_anon = binarize_target(df_anon[config.TARGET_COLUMN]).values
    X_anon = df_anon.drop(columns=[config.TARGET_COLUMN])
    X_test = df_test_raw.drop(columns=[config.TARGET_COLUMN])

    X_anon_array, X_test_array = joint_ohe(X_anon, X_test)
    scaler = StandardScaler().fit(X_anon_array)

    model = LogisticRegression(
        max_iter=1000,
        random_state=config.RANDOM_STATE,
        class_weight="balanced",
    )
    model.fit(scaler.transform(X_anon_array), y_anon)
    predictions = model.predict(scaler.transform(X_test_array))

    y_test = binarize_target(df_test_raw[config.TARGET_COLUMN]).values
    df_test_indexed = df_test_raw.reset_index(drop=True)

    rows: List[Dict] = []
    for race_value, subset in df_test_indexed.groupby("race"):
        positions = subset.index.values
        rows.append({
            "k": k_value,
            "race": race_value,
            "n": len(positions),
            "Accuracy": accuracy_score(y_test[positions], predictions[positions]),
            "F1": f1_score(y_test[positions], predictions[positions], average="weighted"),
        })
    return pd.DataFrame(rows)


def fairness_per_race_dp(
    X_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    races_test: np.ndarray,
    data_norm: float,
    epsilons: List[float] = None,
) -> pd.DataFrame:
    """Calcula la Accuracy por raza para LR-DP a lo largo del barrido de ε.
    La fase 5 lo reutiliza sobre el train k-anonimizado con su barrido reducido."""
    epsilons = epsilons or config.EPSILON_VALUES

    rows: List[Dict] = []
    seeds = config.DP_SEEDS[: config.N_REPETITIONS_DP]
    for epsilon in epsilons:
        for rep, seed in enumerate(seeds):
            model = dpm.LogisticRegression(
                epsilon=epsilon,
                data_norm=data_norm,
                max_iter=100,
                random_state=seed,
            )
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)
            for race_value in np.unique(races_test):
                mask = races_test == race_value
                rows.append({
                    "epsilon": epsilon,
                    "rep": rep,
                    "race": race_value,
                    "n": int(mask.sum()),
                    "Accuracy": float(accuracy_score(y_test[mask], predictions[mask])),
                })
    return pd.DataFrame(rows)
