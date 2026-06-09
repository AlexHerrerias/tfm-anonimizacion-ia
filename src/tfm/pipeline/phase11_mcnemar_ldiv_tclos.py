"""Fase 11 — Test de McNemar sobre las configuraciones extremas de l-div / t-closeness.

Replica la batería de tests McNemar de la Fase 7 sobre la selección de
configuraciones de `config.MCNEMAR_LT_CONFIGURATIONS` (l=5, t=0.5, t=0.25 y
triple Hard). El resto de configuraciones de la Fase 8 se omite por
redundancia con singles ya testeados o por tener un efecto por debajo del
umbral de detección útil.
"""

from typing import Dict, List

import pandas as pd

from tfm import config
from tfm.arx_io import load_arx_arrays
from tfm.data_loader import load_clean_reduced
from tfm.models import build_baseline_models, predict_baselines
from tfm.preprocessing import fit_scaler, raw_frames_from_split, stratified_split
from tfm.statistical_tests import apply_bonferroni, mcnemar_test


def _predict_on_anonymized(filepath, df_test_raw) -> Dict[str, "pd.Series"]:
    """Entrena cada modelo baseline sobre el conjunto anonimizado y devuelve predicciones."""
    arrays = load_arx_arrays(filepath, df_test_raw)
    predictions: Dict[str, "pd.Series"] = {}
    for name, model in build_baseline_models().items():
        model.fit(arrays.X_anon_scaled, arrays.y_anon)
        predictions[name] = model.predict(arrays.X_test_scaled)
    return predictions


def run() -> None:
    config.ensure_dirs()

    df = load_clean_reduced()
    X_train, X_test, y_train, y_test = stratified_split(df)
    # Baseline (k=0): mismo espacio de features que en las Fases 1 y 7 — OHE
    # producido por stratified_split (admission_type_id/time_in_hospital como
    # int) y StandardScaler ajustado sobre el train. Esto garantiza la
    # coherencia con `resultados_kanon.csv` y con `tab:mcnemar`.
    scaler = fit_scaler(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # df_test_raw: copia con admission_type_id/time_in_hospital convertidos a
    # string, requerida por las jerarquías de ARX y por joint_ohe en las
    # configuraciones anonimizadas (compatible con la Fase 7).
    _, df_test_raw = raw_frames_from_split(df, X_train.index, X_test.index)
    y_test_array = y_test.values

    print("Predicciones baseline (k=0)...")
    baseline_predictions = predict_baselines(X_train_scaled, X_test_scaled, y_train)

    rows: List[Dict] = []
    for label, filename in config.MCNEMAR_LT_CONFIGURATIONS:
        filepath = config.ARX_OUTPUTS_DIR / filename
        if not filepath.exists():
            print(f"Aviso: no se encuentra {filepath}, se omite {label}")
            continue
        print(f"  → {label}")
        anon_predictions = _predict_on_anonymized(filepath, df_test_raw)
        for model_name in baseline_predictions:
            p_value, b, c, chi_squared = mcnemar_test(
                y_test_array,
                baseline_predictions[model_name],
                anon_predictions[model_name],
            )
            rows.append({
                "configuracion": label,
                "archivo": filename,
                "modelo": model_name,
                "b": b,
                "c": c,
                "chi2": round(chi_squared, 4),
                "p_value": p_value,
                "significativo_005": p_value < 0.05,
            })

    # Aplicamos Bonferroni sobre la familia completa de tests
    # (N = nº de filas = 4 configuraciones × 4 modelos = 16) para mantener la
    # coherencia con la batería de la Fase 7 (build_mcnemar_table).
    df_results = apply_bonferroni(pd.DataFrame(rows))
    df_results.to_csv(config.RESULTS_LDIV_TCLOS_DIR / "mcnemar_ldiv_tclos.csv", index=False)
    print(f"\n✓ mcnemar_ldiv_tclos.csv guardado con {len(df_results)} filas")
    print(df_results.round(4).to_string(index=False))


if __name__ == "__main__":
    run()
