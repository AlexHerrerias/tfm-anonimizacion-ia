"""Soporte para k-anonimidad con ARX: exportación de CSVs, jerarquías
de generalización y evaluación post-anonimización.
"""

from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from tfm import config
from tfm.arx_io import load_arx_arrays
from tfm.models import build_baseline_models
from tfm.preprocessing import binarize_target


# Jerarquías estáticas — coinciden con las descritas en la memoria
RACE_HIERARCHY = """Caucasian;Caucasian;White;*
AfricanAmerican;AfricanAmerican;NonWhite;*
Hispanic;Hispanic;NonWhite;*
Asian;Asian;NonWhite;*
Other;Other;NonWhite;*"""

GENDER_HIERARCHY = """Female;*
Male;*
Unknown/Invalid;*"""

AGE_HIERARCHY = "\n".join([
    "[0-10);[0-20);[0-40);*",
    "[10-20);[0-20);[0-40);*",
    "[20-30);[20-40);[0-40);*",
    "[30-40);[20-40);[0-40);*",
    "[40-50);[40-60);[40-80);*",
    "[50-60);[40-60);[40-80);*",
    "[60-70);[60-80);[40-80);*",
    "[70-80);[60-80);[40-80);*",
    "[80-90);[80-100);[80-100);*",
    "[90-100);[80-100);[80-100);*",
])

ADMISSION_TYPE_HIERARCHY = """1;Urgente;Conocido;*
2;Urgente;Conocido;*
3;Electivo;Conocido;*
4;Newborn;Conocido;*
5;Desconocido;Desconocido;*
6;Desconocido;Desconocido;*
7;Urgente;Conocido;*
8;Desconocido;Desconocido;*"""


def _time_in_hospital_hierarchy() -> str:
    lines = []
    for i in range(1, 15):
        if i in (1, 2):
            l1, l2 = "[1-2]", "[1-7]"
        elif i in (3, 4):
            l1, l2 = "[3-4]", "[1-7]"
        elif i in (5, 6, 7):
            l1, l2 = "[5-7]", "[1-7]"
        elif i in (8, 9, 10):
            l1, l2 = "[8-10]", "[8-14]"
        else:
            l1, l2 = "[11-14]", "[8-14]"
        lines.append(f"{i};{l1};{l2};*")
    return "\n".join(lines)


def export_for_arx(df_reduced: pd.DataFrame, train_index, test_index, output_dir: Path) -> None:
    """Exporta arx_train.csv y arx_test.csv (separador ';'); los QID numéricos
    se convierten a string para que ARX los importe como categóricos.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_train_raw = df_reduced.loc[train_index].copy()
    df_test_raw = df_reduced.loc[test_index].copy()

    for column in config.QID_NUMERIC_AS_STR:
        df_train_raw[column] = df_train_raw[column].astype(str)
        df_test_raw[column] = df_test_raw[column].astype(str)

    df_train_raw.to_csv(output_dir / "arx_train.csv", sep=";", index=False)
    df_test_raw.to_csv(output_dir / "arx_test.csv", sep=";", index=False)


def export_hierarchies(output_dir: Path) -> None:
    """Genera los cinco archivos de jerarquía esperados por ARX Desktop."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "race.csv": RACE_HIERARCHY,
        "gender.csv": GENDER_HIERARCHY,
        "age.csv": AGE_HIERARCHY,
        "admission_type_id.csv": ADMISSION_TYPE_HIERARCHY,
        "time_in_hospital.csv": _time_in_hospital_hierarchy(),
    }
    for filename, content in files.items():
        (output_dir / filename).write_text(content)


def evaluate_kanon(filepath_anon: Path, k_value: int, df_test_raw: pd.DataFrame) -> pd.DataFrame:
    """Evalúa un CSV anonimizado por ARX con los 4 modelos; OHE conjunto
    sobre [train_anon ∪ test_raw] para igualar el column space.
    """
    arrays = load_arx_arrays(filepath_anon, df_test_raw)
    y_test = binarize_target(df_test_raw[config.TARGET_COLUMN]).values

    rows: List[Dict] = []
    for name, model in build_baseline_models().items():
        model.fit(arrays.X_anon_scaled, arrays.y_anon)
        predictions = model.predict(arrays.X_test_scaled)
        rows.append({
            "Modelo": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "F1-Score": f1_score(y_test, predictions, average="weighted"),
            "k": k_value,
            "pct_suprimido": round(arrays.suppression_pct, 2),
            "filas_efectivas": arrays.n_rows,
            "columnas_OHE": arrays.n_ohe_columns,
        })
    return pd.DataFrame(rows)
