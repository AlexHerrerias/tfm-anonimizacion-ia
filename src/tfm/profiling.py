"""Perfilado de privacidad: riesgo basal de reidentificación del dataset sin
anonimizar, según el tamaño de las clases de equivalencia de los cinco QIDs.
"""

from typing import Dict, List

import pandas as pd

from tfm import config


def privacy_profile(df: pd.DataFrame, qids: List[str] = None) -> Dict:
    """Perfil de riesgo basal agrupando por QIDs; idéntico antes o después de
    la reducción CIE-9 (no altera filas ni QIDs).
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
