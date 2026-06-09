"""Perfilado de privacidad: riesgo basal de reidentificación.

Cuantifica el riesgo de enlace del dataset SIN anonimizar mediante el
tamaño de las clases de equivalencia inducidas por los cinco
cuasi-identificadores (QIDs): un registro con clase de tamaño k=1 es
únicamente identificable por un atacante con acceso a registros
administrativos.

Estos números respaldan la motivación de privacidad de la memoria
(Capítulo 3).
"""

from typing import Dict, List

import pandas as pd

from tfm import config


def privacy_profile(df: pd.DataFrame, qids: List[str] = None) -> Dict:
    """Calcula el perfil de riesgo basal de reidentificación.

    Las clases de equivalencia se construyen agrupando por los QIDs sobre
    el dataset limpio (la reducción CIE-9 no altera filas ni QIDs, por lo
    que el resultado es idéntico antes o después de ella).
    """
    if qids is None:
        qids = config.QID_COLUMNS
    equivalence_classes = df.groupby(qids).size().reset_index(name="size")

    n_total = len(df)
    n_unique = int(equivalence_classes[equivalence_classes["size"] == 1]["size"].sum())
    n_vulnerable_k5 = int(equivalence_classes[equivalence_classes["size"] < 5]["size"].sum())

    return {
        "n_total": n_total,
        "n_equivalence_classes": int(len(equivalence_classes)),
        "n_k1": n_unique,
        "pct_k1": round(n_unique / n_total * 100, 2),
        "n_k_lt5": n_vulnerable_k5,
        "pct_k_lt5": round(n_vulnerable_k5 / n_total * 100, 2),
    }
