"""Aplicación de Privacidad Diferencial mediante diffprivlib."""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from tfm import config
from tfm.arx_io import load_arx_arrays
from tfm.models import build_dp_models
from tfm.preprocessing import (
    binarize_target,
    percentile_bounds,
    percentile_data_norm,
)


def sweep_dp(
    X_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    y_train,
    y_test,
    data_norm: float,
    bounds,
    epsilons: List[float] = None,
) -> pd.DataFrame:
    """Barrido de ε x repeticiones x 2 modelos; una fila por (modelo, epsilon, repetición)."""
    epsilons = epsilons or config.EPSILON_VALUES
    seeds = config.DP_SEEDS[: config.N_REPETITIONS_DP]

    rows: List[Dict] = []
    for epsilon in epsilons:
        for rep, seed in enumerate(seeds):
            for model_name, model in build_dp_models(epsilon, data_norm, bounds, seed).items():
                model.fit(X_train_scaled, y_train)
                predictions = model.predict(X_test_scaled)
                rows.append({
                    "modelo": model_name,
                    "epsilon": epsilon,
                    "rep": rep,
                    "seed": seed,
                    "Accuracy": accuracy_score(y_test, predictions),
                    "F1-Score": f1_score(y_test, predictions, average="weighted"),
                })
    return pd.DataFrame(rows)


def dp_on_kanonimized(
    filepath_kanon: Path,
    df_test_raw: pd.DataFrame,
) -> pd.DataFrame:
    """Sweep DP sobre el train k-anonimizado de ARX (sin filas suprimidas),
    para medir el coste de utilidad de la defensa en profundidad k-anon + DP.
    """
    epsilons = [0.1, 1.0, 10.0]  # barrido reducido de la fase combo

    arrays = load_arx_arrays(filepath_kanon, df_test_raw)

    data_norm = percentile_data_norm(arrays.X_anon_scaled, percentile=95)
    bounds = percentile_bounds(arrays.X_anon_scaled)
    y_test = binarize_target(df_test_raw[config.TARGET_COLUMN]).values

    return sweep_dp(
        arrays.X_anon_scaled,
        arrays.X_test_scaled,
        arrays.y_anon,
        y_test,
        data_norm,
        bounds,
        epsilons=epsilons,
    )
