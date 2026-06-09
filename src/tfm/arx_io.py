"""Carga unificada de los CSVs anonimizados exportados por ARX Desktop.

Centraliza el bloque común a todas las fases de evaluación: leer el CSV
(sep=';'), eliminar las filas suprimidas, binarizar el target, aplicar el
OHE conjunto con el test en claro y estandarizar.

Semántica de supresión: ARX exporta los registros suprimidos con todos los
QID a `*`. El detector activo es la heurística `race == '*'` (la jerarquía
de `race` solo alcanza `*` en su nivel máximo, que coincide con la
supresión), la misma con la que se calcularon los resultados de la memoria.
`mask_suppressed_rows` implementa la verificación completa (todos los QIDs
a `*`); sobre los CSVs de este TFM ambas coinciden al 100 %, pero la
verificación completa queda blindada frente a futuros cambios en las
jerarquías.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from tfm import config
from tfm.preprocessing import binarize_target, joint_ohe


def read_arx_csv(filepath: Path) -> Tuple[pd.DataFrame, int]:
    """Lee un CSV exportado por ARX y elimina las filas suprimidas.

    Devuelve (df_sin_suprimidas, n_filas_iniciales) para poder calcular
    el porcentaje de supresión.
    """
    df_anon = pd.read_csv(filepath, sep=";")
    n_initial = len(df_anon)
    mask_suppressed = df_anon["race"].astype(str).str.fullmatch(r"\*")
    return df_anon[~mask_suppressed].copy(), n_initial


def mask_suppressed_rows(df: pd.DataFrame, qids: list = None) -> pd.Series:
    """Detector de supresión completo: todos los QIDs a `*`.

    Alternativa robusta a la heurística `race == '*'` usada por
    `read_arx_csv`. Sobre los CSVs de este TFM ambas coinciden al 100 %.
    """
    if qids is None:
        qids = config.QID_COLUMNS
    available = [c for c in qids if c in df.columns]
    return df[available].astype(str).eq("*").all(axis=1)


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
    """Pipeline completo CSV de ARX → matrices escaladas.

    1. Lee el CSV (sep=';') y elimina filas suprimidas (`read_arx_csv`).
    2. Binariza el target y separa features.
    3. OHE conjunto [train_anon ∪ test_raw] para igualar el column space.
    4. StandardScaler ajustado sobre el train anonimizado, aplicado a ambos.
    5. Si `dtype` se especifica (p. ej. np.float32 para la auditoría MIA de
       la Fase 9), castea las matrices después de escalar.
    """
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
