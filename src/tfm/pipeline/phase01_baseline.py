"""Fase 1 — Establece el baseline de utilidad sin privacidad.

Entrena los cuatro modelos de referencia sobre el conjunto Diabetes 130-US
hospitals y guarda las métricas en arx_kit/results/kanon/baseline.csv.
"""

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from tfm import config
from tfm.data_loader import load_clean_reduced
from tfm.models import build_baseline_models
from tfm.preprocessing import fit_scaler, stratified_split


def run() -> None:
    config.ensure_dirs()

    df = load_clean_reduced()
    print(f"Dataset tras limpieza y reducción: {df.shape}")

    X_train, X_test, y_train, y_test = stratified_split(df)
    scaler = fit_scaler(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rows = []
    for name, model in build_baseline_models().items():
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        rows.append({
            "Modelo": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "F1-Score": f1_score(y_test, predictions, average="weighted"),
            "k": 0,
        })

    df_baseline = pd.DataFrame(rows).sort_values("F1-Score", ascending=False)
    output_path = config.RESULTS_KANON_DIR / "baseline.csv"
    df_baseline.to_csv(output_path, index=False)
    print("\nBaseline:")
    print(df_baseline.round(4).to_string(index=False))
    print(f"\nGuardado en {output_path}")


if __name__ == "__main__":
    run()
