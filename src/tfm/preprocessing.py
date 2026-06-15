"""División train/test, codificación One-Hot y estandarización."""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tfm import config


def load_baselines() -> Dict[str, Tuple[float, float]]:
    """Lee los baselines (k=0) de `resultados_kanon.csv` y devuelve
    {modelo: (accuracy, f1)} redondeado a 4 decimales."""
    csv_path = config.RESULTS_KANON_DIR / "resultados_kanon.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No se encuentra {csv_path}. Ejecuta primero `tfm run 03` "
            "para generar los baselines."
        )
    df = pd.read_csv(csv_path)
    baseline = df[df["k"] == 0]
    return {
        row["Modelo"]: (round(float(row["Accuracy"]), 4), round(float(row["F1-Score"]), 4))
        for _, row in baseline.iterrows()
    }


def binarize_target(series: pd.Series) -> pd.Series:
    """Convierte la variable readmitted en clasificación binaria 0/1."""
    return (series != "NO").astype(int)


def stratified_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Partición estratificada con semilla de config; el OHE se aplica sobre
    el conjunto completo antes del split para alinear columnas."""
    X_raw = df.drop(columns=[config.TARGET_COLUMN])
    y_bin = binarize_target(df[config.TARGET_COLUMN])
    X_encoded = pd.get_dummies(X_raw, drop_first=True)

    return train_test_split(
        X_encoded,
        y_bin,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y_bin,
    )


def raw_frames_from_split(
    df: pd.DataFrame, train_index, test_index, coerce_qids: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (df_train_raw, df_test_raw) alineados con `stratified_split`;
    con `coerce_qids` los QID numéricos pasan a string (formato ARX)."""
    df_train_raw = df.loc[train_index].copy()
    df_test_raw = df.loc[test_index].copy()
    if coerce_qids:
        for column in config.QID_NUMERIC_AS_STR:
            df_train_raw[column] = df_train_raw[column].astype(str)
            df_test_raw[column] = df_test_raw[column].astype(str)
    return df_train_raw, df_test_raw


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Ajusta un StandardScaler sobre el conjunto de entrenamiento."""
    return StandardScaler().fit(X_train)


def joint_ohe(X_anon: pd.DataFrame, X_test_raw: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """OHE conjunto train anonimizado + test para igualar el column space tras
    la generalización. No es data leakage: no aprende estadísticos, solo
    determina columnas (justificación completa en la memoria)."""
    n_anon = len(X_anon)
    stacked = pd.concat([X_anon, X_test_raw], ignore_index=True)
    encoded = pd.get_dummies(stacked, drop_first=True)
    return encoded.iloc[:n_anon].values, encoded.iloc[n_anon:].values


def percentile_data_norm(X_scaled: np.ndarray, percentile: int = 95) -> float:
    """Percentil de las normas L2 por fila: cota recortada usada como
    `data_norm` en diffprivlib para no sobrecalibrar el ruido."""
    norms = np.linalg.norm(X_scaled, axis=1)
    return float(np.percentile(norms, percentile))


def percentile_bounds(X_scaled: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Cotas robustas (percentiles 1-99) por característica para diffprivlib.GaussianNB."""
    return np.percentile(X_scaled, 1, axis=0), np.percentile(X_scaled, 99, axis=0)
