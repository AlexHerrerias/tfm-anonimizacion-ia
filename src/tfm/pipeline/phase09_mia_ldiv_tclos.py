"""Fase 9 — Auditoría MIA Black-Box sobre las configuraciones l-div / t-clos más estrictas.

Audita los cuatro escenarios de `config.MIA_LT_TARGETS` sobre Regresión
Logística (k=5 referencia, l=5, t=0.25 y triple Hard).

Resultado: tabla mia_ldiv_tclos.csv con TPR, FPR y Advantage por configuración.

Nota de reproducibilidad: se utiliza un RNG independiente por configuración
(`np.random.default_rng(seed)`), con la semilla fijada explícitamente a
partir de `config.DP_SEEDS` por índice de escenario. Esto elimina la
dependencia del orden de las configuraciones que tenía la versión inicial
basada en `np.random.seed` global, y permite añadir o reordenar
configuraciones sin alterar los muestreos de las demás.
"""

from typing import List

import numpy as np
import pandas as pd

from tfm import config
from tfm.arx_io import load_arx_arrays
from tfm.data_loader import load_clean_reduced
from tfm.mia import mia_attack_rates
from tfm.preprocessing import raw_frames_from_split, stratified_split


def run() -> None:
    config.ensure_dirs()

    df = load_clean_reduced()
    X_train, X_test, y_train, y_test = stratified_split(df)
    _, df_test_raw = raw_frames_from_split(df, X_train.index, X_test.index)
    y_test_array = y_test.values

    results: List[dict] = []
    for i, (label, filename) in enumerate(config.MIA_LT_TARGETS):
        filepath = config.ARX_OUTPUTS_DIR / filename
        if not filepath.exists():
            print(f"Aviso: no se encuentra {filepath}, se omite")
            continue
        arrays = load_arx_arrays(filepath, df_test_raw, dtype=np.float32)
        seed = config.DP_SEEDS[i % len(config.DP_SEEDS)]
        tpr, fpr, advantage, auc = mia_attack_rates(
            arrays.X_anon_scaled, arrays.y_anon, arrays.X_test_scaled, y_test_array, seed=seed
        )
        results.append({
            "escenario": label,
            "archivo": filename,
            "seed": seed,
            "TPR": round(tpr, 4),
            "FPR": round(fpr, 4),
            "Advantage": round(advantage, 4),
            "AUC": round(auc, 4),
        })
        print(f"  ✓ {label} (seed={seed})")

    df_mia = pd.DataFrame(results)
    df_mia.to_csv(config.RESULTS_MIA_DIR / "mia_ldiv_tclos.csv", index=False)
    print(df_mia.round(4).to_string(index=False))


if __name__ == "__main__":
    run()
