"""Fase 5 — Defensa en profundidad: DP aplicada sobre k=10.

Replica el sweep ε ∈ {0.1, 1, 10} sobre el conjunto previamente
anonimizado con k=10 para evaluar si la combinación neutraliza el
efecto adverso de la DP sobre los grupos minoritarios.
"""

from tfm import config
from tfm.data_loader import load_clean_reduced
from tfm.differential_privacy import dp_on_kanonimized
from tfm.preprocessing import raw_frames_from_split, stratified_split


def run() -> None:
    config.ensure_dirs()

    df = load_clean_reduced()
    X_train, X_test, _, _ = stratified_split(df)
    _, df_test_raw = raw_frames_from_split(df, X_train.index, X_test.index)

    filepath = config.ARX_OUTPUTS_DIR / config.arx_output_filename(10)
    if not filepath.exists():
        raise FileNotFoundError(
            f"No se encuentra {filepath}. Ejecuta antes la fase ARX para k=10."
        )

    df_combo = dp_on_kanonimized(
        filepath_kanon=filepath,
        df_test_raw=df_test_raw,
    )
    df_combo.to_csv(config.RESULTS_DP_DIR / "resultados_combo.csv", index=False)

    aggregated = df_combo.groupby(["modelo", "epsilon"])["Accuracy"].agg(["mean", "std"]).round(4)
    print("Resultados de la combinación k=10 + DP (Accuracy mean ± std):")
    print(aggregated.to_string())


if __name__ == "__main__":
    run()
