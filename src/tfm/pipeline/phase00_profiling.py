"""Fase 0 — Perfilado de privacidad: riesgo basal de reidentificación del dataset
limpio sin anonimizar. Salida: arx_kit/results/kanon/perfil_privacidad.csv
"""

import pandas as pd

from tfm import config
from tfm.data_loader import load_clean_reduced
from tfm.profiling import privacy_profile


def run() -> None:
    config.ensure_dirs()

    df = load_clean_reduced()
    profile = privacy_profile(df)

    output_path = config.RESULTS_KANON_DIR / "perfil_privacidad.csv"
    pd.DataFrame([profile]).to_csv(output_path, index=False)

    print(f"Combinaciones únicas de QIDs: {profile['n_equivalence_classes']}")
    print(
        f"Pacientes con k=1 (vulnerabilidad extrema): {profile['n_k1']} "
        f"({profile['pct_k1']:.2f} %)"
    )
    print(f"Pacientes con k<5: {profile['n_k_lt5']} ({profile['pct_k_lt5']:.2f} %)")
    print(f"\nGuardado en {output_path}")


if __name__ == "__main__":
    run()
