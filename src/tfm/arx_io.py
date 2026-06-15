"""Carga unificada de los CSVs anonimizados de ARX: lectura, retirada de
filas suprimidas (detector `race == '*'`, el de los resultados de la
memoria), binarización del target, OHE conjunto y estandarización."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from tfm import config
from tfm.preprocessing import binarize_target, joint_ohe


def read_arx_csv(filepath: Path) -> Tuple[pd.DataFrame, int]:
    """Lee un CSV de ARX sin las filas suprimidas; devuelve también el número
    inicial de filas para calcular el porcentaje de supresión."""
    df_anon = pd.read_csv(filepath, sep=";")
    n_initial = len(df_anon)
    mask_suppressed = df_anon["race"].astype(str).str.fullmatch(r"\*")
    return df_anon[~mask_suppressed].copy(), n_initial


@dataclass
class ArxArrays:
    """Matrices listas para entrenar/evaluar a partir de un export de ARX."""

    X_anon_scaled: np.ndarray
    y_anon: np.ndarray
    X_test_scaled: np.ndarray
    suppression_pct: float
    n_rows: int            # filas efectivas tras eliminar suprimidas
    n_ohe_columns: int     # dimensión del espacio OHE conjunto


def load_arx_arrays(
    filepath: Path,
    df_test_raw: pd.DataFrame,
    dtype=None,
) -> ArxArrays:
    """CSV de ARX a matrices escaladas: filtra suprimidas, binariza el target,
    OHE conjunto con `df_test_raw` y StandardScaler ajustado sobre el train
    anonimizado; `dtype` castea tras escalar (p. ej. float32 en la Fase 9)."""
    df_anon, n_initial = read_arx_csv(filepath)
    suppression_pct = (n_initial - len(df_anon)) / n_initial * 100

    y_anon = binarize_target(df_anon[config.TARGET_COLUMN]).values
    X_anon = df_anon.drop(columns=[config.TARGET_COLUMN])
    X_test = df_test_raw.drop(columns=[config.TARGET_COLUMN])

    X_anon_array, X_test_array = joint_ohe(X_anon, X_test)
    scaler = StandardScaler().fit(X_anon_array)
    X_anon_scaled = scaler.transform(X_anon_array)
    X_test_scaled = scaler.transform(X_test_array)
    if dtype is not None:
        X_anon_scaled = X_anon_scaled.astype(dtype)
        X_test_scaled = X_test_scaled.astype(dtype)

    return ArxArrays(
        X_anon_scaled=X_anon_scaled,
        y_anon=y_anon,
        X_test_scaled=X_test_scaled,
        suppression_pct=suppression_pct,
        n_rows=len(df_anon),
        n_ohe_columns=X_anon_array.shape[1],
    )
