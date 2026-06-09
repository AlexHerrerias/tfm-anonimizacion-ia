"""Operaciones de soporte para la extensión a l-diversidad y t-closeness.

Reutiliza la lógica de evaluación de k-anonimidad y añade:
- Verificación de las restricciones l-diversity / t-closeness sobre el CSV
  exportado por ARX (post-hoc).
- Cálculo de la Earth Mover's Distance con métrica Equal (equivalente a la
  mitad de la distancia L1) entre la distribución del atributo sensible de
  cada clase de equivalencia y la distribución global del conjunto.

Las configuraciones se exportan manualmente desde ARX Desktop con el
atributo sensible (`config.SENSITIVE_ATTRIBUTE`) marcado como Sensitive y
los QIDs idénticos a la fase de k-anonimidad. Esta utilidad asume que los
CSV están en `config.ARX_OUTPUTS_DIR`.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from tfm import config
from tfm.arx_io import load_arx_arrays, read_arx_csv
from tfm.models import build_baseline_models
from tfm.preprocessing import binarize_target, joint_ohe


def _global_sa_distribution(df_train_raw: pd.DataFrame) -> pd.Series:
    """Distribución de la SA en el conjunto de entrenamiento sin anonimizar."""
    return df_train_raw[config.SENSITIVE_ATTRIBUTE].value_counts(normalize=True)


def verify_constraints(
    filepath: Path,
    qid_columns: List[str],
    df_train_raw: pd.DataFrame,
    expected_k: int = 5,
    expected_l: Optional[int] = None,
    expected_t: Optional[float] = None,
) -> Dict:
    """Carga un CSV anonimizado y verifica las restricciones declaradas.

    Devuelve un diccionario con métricas estructurales (clases de equivalencia,
    supresión, l-diversidad observada y EMD máxima) y los flags de cumplimiento.
    """
    df_clean, n_initial = read_arx_csv(filepath)
    suppression = (n_initial - len(df_clean)) / n_initial * 100

    global_dist = _global_sa_distribution(df_train_raw)
    sa_cats = sorted(global_dist.index)
    g = np.array([global_dist.get(c, 0.0) for c in sa_cats])

    df_clean["_eq_class"] = df_clean[qid_columns].astype(str).agg("|".join, axis=1)
    crosstab = pd.crosstab(df_clean["_eq_class"], df_clean[config.SENSITIVE_ATTRIBUTE])
    for category in sa_cats:
        if category not in crosstab.columns:
            crosstab[category] = 0
    crosstab = crosstab[sa_cats]
    class_sizes = crosstab.sum(axis=1).values
    class_proba = crosstab.div(crosstab.sum(axis=1), axis=0).values
    emds = 0.5 * np.abs(class_proba - g[None, :]).sum(axis=1)
    distinct_sa = (crosstab > 0).sum(axis=1).values

    k_ok = bool((class_sizes >= expected_k).all())
    l_ok = (
        None if expected_l is None else bool((distinct_sa >= expected_l).all())
    )
    t_ok = (
        None if expected_t is None else bool((emds <= expected_t + 1e-6).all())
    )

    return {
        "filepath": str(filepath),
        "n_filas": int(len(df_clean)),
        "suppression_pct": round(float(suppression), 2),
        "n_classes": int(len(crosstab)),
        "size_min": int(class_sizes.min()),
        "size_median": int(np.median(class_sizes)),
        "l_min_observed": int(distinct_sa.min()),
        "l_median_observed": int(np.median(distinct_sa)),
        "emd_max": round(float(emds.max()), 4),
        "k_satisfied": k_ok,
        "l_satisfied": l_ok,
        "t_satisfied": t_ok,
    }


def evaluate_ldiv_tclos(
    filepath_anon: Path,
    df_test_raw: pd.DataFrame,
    tag: Dict,
) -> pd.DataFrame:
    """Entrena los modelos baseline sobre un CSV anonimizado y devuelve métricas.

    `tag` debe contener al menos las claves: `k`, `l`, `t`, `tipo`, `archivo`.
    Mantiene la misma interfaz que `kanon.evaluate_kanon` para que las funciones
    de plotting puedan reutilizarse.
    """
    arrays = load_arx_arrays(filepath_anon, df_test_raw)
    y_test = binarize_target(df_test_raw[config.TARGET_COLUMN]).values

    rows: List[Dict] = []
    for name, model in build_baseline_models().items():
        model.fit(arrays.X_anon_scaled, arrays.y_anon)
        preds = model.predict(arrays.X_test_scaled)
        rows.append({
            **tag,
            "Modelo": name,
            "Accuracy": accuracy_score(y_test, preds),
            "F1-Score": f1_score(y_test, preds, average="weighted"),
            "pct_suprimido": round(arrays.suppression_pct, 2),
            "filas_efectivas": arrays.n_rows,
            "columnas_OHE": arrays.n_ohe_columns,
        })
    return pd.DataFrame(rows)


def fairness_ldiv_tclos(
    filepath_anon: Optional[Path],
    df_train_raw: pd.DataFrame,
    df_test_raw: pd.DataFrame,
    tag: Dict,
    min_subgroup_size: int = 10,
) -> pd.DataFrame:
    """Calcula Accuracy de la Regresión Logística por subgrupo `race`.

    Si `filepath_anon` es None se utiliza el conjunto de entrenamiento sin
    anonimizar (referencia baseline).
    """
    if filepath_anon is None:
        df_train = df_train_raw.copy()
    else:
        df_train, _ = read_arx_csv(filepath_anon)

    y_train = binarize_target(df_train[config.TARGET_COLUMN]).values
    X_train = df_train.drop(columns=[config.TARGET_COLUMN])
    X_test = df_test_raw.drop(columns=[config.TARGET_COLUMN])

    X_train_array, X_test_array = joint_ohe(X_train, X_test)
    scaler = StandardScaler().fit(X_train_array)
    model = LogisticRegression(
        max_iter=1000, random_state=config.RANDOM_STATE, class_weight="balanced"
    )
    model.fit(scaler.transform(X_train_array), y_train)
    preds = model.predict(scaler.transform(X_test_array))
    y_test = binarize_target(df_test_raw[config.TARGET_COLUMN]).values

    rows: List[Dict] = []
    for race in df_test_raw["race"].unique():
        mask = (df_test_raw["race"] == race).values
        if mask.sum() < min_subgroup_size:
            continue
        rows.append({
            **tag,
            "race": race,
            "n": int(mask.sum()),
            "Accuracy": accuracy_score(y_test[mask], preds[mask]),
        })
    return pd.DataFrame(rows)
