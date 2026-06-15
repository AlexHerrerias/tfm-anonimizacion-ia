"""Fase 2 — Exporta CSVs y jerarquías a arx_kit/inputs/ para ARX Desktop.
La anonimización es manual (docs/guia_arx.md) y sus CSV resultantes deben
guardarse en arx_kit/arx_outputs/.
"""

from tfm import config
from tfm.data_loader import load_clean_reduced
from tfm.kanon import export_for_arx, export_hierarchies
from tfm.preprocessing import stratified_split


def run() -> None:
    df = load_clean_reduced()
    X_train, X_test, _, _ = stratified_split(df)

    export_for_arx(
        df_reduced=df,
        train_index=X_train.index,
        test_index=X_test.index,
        output_dir=config.INPUTS_DIR,
    )
    export_hierarchies(output_dir=config.HIERARCHIES_DIR)

    print(f"CSVs exportados a {config.INPUTS_DIR}")
    print(f"Jerarquías exportadas a {config.HIERARCHIES_DIR}")
    print("\nSiguiente paso: importar arx_train.csv en ARX Desktop, cargar")
    print("las jerarquías, ejecutar el barrido k ∈ {2, 5, 10, 25, 50} y")
    print(f"exportar cada resultado como {config.ARX_OUTPUTS_DIR}/arx_output_k<k>.csv (sep=';').")


if __name__ == "__main__":
    run()
