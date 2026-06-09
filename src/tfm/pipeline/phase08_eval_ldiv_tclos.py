"""Fase 8 — Evaluación de l-diversidad, t-closeness y combinaciones triples.

Lee los CSVs anonimizados exportados manualmente desde ARX Desktop
(con `diag_1_category` marcado como Sensitive y los mismos QIDs de la
fase de k-anonimidad). Para cada configuración de `config.LT_CONFIGURATIONS`
entrena los cuatro modelos baseline, calcula utilidad sobre el conjunto de
test, equidad por subgrupos race y verifica las restricciones de privacidad
declaradas.
"""

import pandas as pd

from tfm import config
from tfm.data_loader import load_clean_reduced
from tfm.ldiv_tclos import (
    evaluate_ldiv_tclos,
    fairness_ldiv_tclos,
    verify_constraints,
)
from tfm.preprocessing import raw_frames_from_split, stratified_split


def run() -> None:
    config.ensure_dirs()

    df = load_clean_reduced()
    X_train, X_test, _, _ = stratified_split(df)
    df_train_raw, df_test_raw = raw_frames_from_split(df, X_train.index, X_test.index)

    utility_parts = []
    fairness_parts = [fairness_ldiv_tclos(
        None, df_train_raw, df_test_raw,
        {"k": 5, "l": None, "t": None, "tipo": "k_solo", "archivo": "baseline"},
    )]
    verifications = []

    for filename, k, l, t, tipo in config.LT_CONFIGURATIONS:
        filepath = config.ARX_OUTPUTS_DIR / filename
        if not filepath.exists():
            print(f"Aviso: no se encuentra {filepath}, se omite")
            continue
        tag = {"k": k, "l": l, "t": t, "tipo": tipo, "archivo": filename}
        utility_parts.append(evaluate_ldiv_tclos(filepath, df_test_raw, tag))
        fairness_parts.append(fairness_ldiv_tclos(
            filepath, df_train_raw, df_test_raw, tag,
        ))
        verifications.append(verify_constraints(
            filepath, config.QID_COLUMNS, df_train_raw,
            expected_k=k, expected_l=l, expected_t=t,
        ))
        print(f"  ✓ {filename}")

    df_utility = pd.concat(utility_parts, ignore_index=True)
    df_utility.to_csv(config.RESULTS_LDIV_TCLOS_DIR / "resultados_ldiv_tclos.csv", index=False)
    df_fairness = pd.concat(fairness_parts, ignore_index=True)
    df_fairness.to_csv(config.RESULTS_LDIV_TCLOS_DIR / "fairness_ldiv_tclos.csv", index=False)
    pd.DataFrame(verifications).to_csv(
        config.RESULTS_LDIV_TCLOS_DIR / "verificacion_ldiv_tclos.csv", index=False
    )
    print("\nGuardados:")
    print(f"  - resultados_ldiv_tclos.csv ({len(df_utility)} filas)")
    print(f"  - fairness_ldiv_tclos.csv ({len(df_fairness)} filas)")
    print(f"  - verificacion_ldiv_tclos.csv ({len(verifications)} filas)")


if __name__ == "__main__":
    run()
