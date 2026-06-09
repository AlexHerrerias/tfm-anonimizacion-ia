"""División train/test, codificación One-Hot y estandarización."""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tfm import config


def load_baselines(csv_path: Path = None) -> Dict[str, Tuple[float, float]]:
    """Lee los baselines (k=0) de `resultados_kanon.csv` y devuelve un dict.

    Devuelve {modelo: (accuracy, f1_score)} para cada modelo del baseline,
    redondeado a 4 decimales (consistente con la precisión de las tablas
    de la memoria). Si el CSV no existe, lanza FileNotFoundError con un
    mensaje claro.

    Esta función centraliza el acceso a los baselines y evita que los
    valores aparezcan hardcoded en múltiples fases.
    """
    if csv_path is None:
        csv_path = config.RESULTS_KANON_DIR / "resultados_kanon.csv"
    csv_path = Path(csv_path)
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
    """Partición estratificada con la semilla fijada en config.

    Devuelve (X_train_encoded, X_test_encoded, y_train, y_test) donde
    las dos primeras son matrices OHE aplicadas sobre el conjunto completo
    antes del split para garantizar columnas alineadas.
    """
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
    """Devuelve (df_train_raw, df_test_raw) alineados con `stratified_split`.

    Reconstruye los DataFrames originales (sin OHE) de train y test a partir
    de los índices del split estratificado. Si `coerce_qids` es True, los
    QIDs numéricos se convierten a string (formato que exportan las
    jerarquías de ARX, necesario para el OHE conjunto con CSVs anonimizados).
    """
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
    """Aplica One-Hot Encoding conjunto sobre la unión train_anon ∪ test.

    Garantiza un column space idéntico entre el conjunto anonimizado
    y el conjunto de prueba en claro. Necesario para evaluar modelos
    entrenados sobre datos generalizados frente a datos originales.

    Justificación metodológica
    --------------------------
    Esta función concatena train (anonimizado) y test (en claro) antes del OHE.
    A primera vista puede parecer *data leakage* informativo, pero NO lo es:

    1. El OHE es una operación libre de parámetros estadísticos. No aprende
       medias, varianzas ni etiquetas: solo determina el conjunto de columnas.
    2. El motivo del OHE conjunto es estructural: tras la generalización ARX
       el train contiene valores agregados (ej. age=`[40-60)`) mientras que
       el test conserva valores originales (ej. age=`[40-50)`). Sin un OHE
       compartido, los espacios de columnas son incompatibles y el modelo
       entrenado no podría puntuar el test.
    3. Las columnas exclusivas del test (no presentes en train tras la
       generalización) quedan implícitamente a peso cero en cualquier modelo
       lineal: el algoritmo no ve esos features durante el fit. No hay
       transferencia de información estadística.
    4. La alternativa rigurosa sería aplicar las mismas jerarquías ARX al
       test antes del OHE, lo que cambia el experimento (test también
       anonimizado, evaluación in-distribution en lugar de out-of-distribution
       respecto a la generalización).

    Se documenta esta decisión para que sea reproducible y auditable.
    """
    n_anon = len(X_anon)
    stacked = pd.concat([X_anon, X_test_raw], ignore_index=True)
    encoded = pd.get_dummies(stacked, drop_first=True)
    return encoded.iloc[:n_anon].values, encoded.iloc[n_anon:].values


def percentile_data_norm(X_scaled: np.ndarray, percentile: int = 95) -> float:
    """Devuelve el percentil indicado de las normas L2 de las filas.

    Esta cota recortada se utiliza como `data_norm` en `diffprivlib`
    para evitar la sobrecalibración del ruido por filas atípicas.
    """
    norms = np.linalg.norm(X_scaled, axis=1)
    return float(np.percentile(norms, percentile))


def percentile_bounds(X_scaled: np.ndarray, low: int = 1, high: int = 99) -> Tuple[np.ndarray, np.ndarray]:
    """Cotas robustas por característica para diffprivlib.GaussianNB."""
    return np.percentile(X_scaled, low, axis=0), np.percentile(X_scaled, high, axis=0)
