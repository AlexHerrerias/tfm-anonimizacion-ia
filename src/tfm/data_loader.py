"""Ingesta y limpieza del conjunto Diabetes 130-US hospitals."""

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

from tfm import config


def load_raw_dataset() -> pd.DataFrame:
    """Descarga el dataset UCI con caché local en pickle; borrar
    `data_cache/` fuerza una nueva descarga."""
    if config.UCI_CACHE_FILE.exists():
        return pd.read_pickle(config.UCI_CACHE_FILE)

    dataset = fetch_ucirepo(id=config.DATASET_ID)
    df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_pickle(config.UCI_CACHE_FILE)
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza de la memoria: suprime `weight` (96,85 % nulos), imputa
    'No_Registrado' en nulos clínicos y elimina filas con nulos residuales."""
    df = df.copy()
    df.replace("?", np.nan, inplace=True)
    df.drop(columns=["weight"], inplace=True)
    df[config.IMPUTE_COLUMNS] = df[config.IMPUTE_COLUMNS].fillna(config.IMPUTE_VALUE)
    df.dropna(subset=config.DROP_NA_SUBSET, inplace=True)
    return df


def map_diagnosis(code) -> str:
    """Mapea un código CIE-9 a una de las nueve macrocategorías clínicas."""
    if pd.isna(code) or code == "?":
        return "Other"
    if isinstance(code, str) and (code.startswith("V") or code.startswith("E")):
        return "Other"
    try:
        value = float(code)
    except (TypeError, ValueError):
        return "Other"

    if 250 <= value < 251:
        return "Diabetes"
    if (390 <= value <= 459) or value == 785:
        return "Circulatory"
    if (460 <= value <= 519) or value == 786:
        return "Respiratory"
    if (520 <= value <= 579) or value == 787:
        return "Digestive"
    if (580 <= value <= 629) or value == 788:
        return "Genitourinary"
    if 710 <= value <= 739:
        return "Musculoskeletal"
    if 800 <= value <= 999:
        return "Injury"
    if 140 <= value <= 239:
        return "Neoplasms"
    return "Other"


def reduce_diagnoses(df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa los códigos CIE-9 en 9 macrocategorías (OHE de 2.405 a 121
    dimensiones, condición previa para la Privacidad Diferencial)."""
    df = df.copy()
    for column in ("diag_1", "diag_2", "diag_3"):
        df[f"{column}_category"] = df[column].apply(map_diagnosis)
    return df.drop(columns=config.DROP_COLUMNS_FOR_ANONYMIZATION)


def load_clean_reduced() -> pd.DataFrame:
    """Pipeline completo: descarga, limpieza y agrupación de diagnósticos."""
    df = load_raw_dataset()
    df = clean_dataset(df)
    df = reduce_diagnoses(df)
    return df


def load_split_frames(verbose: bool = True):
    """Devuelve (df_train_raw, df_test_raw) del split canónico; sin acceso a
    UCI/caché lo reconstruye desde arx_kit/inputs (mismas filas, orden y
    dtypes que el canónico). Los QID numéricos se devuelven sin coercionar."""
    from tfm.preprocessing import stratified_split

    try:
        df = load_clean_reduced()
        X_train, X_test, _, _ = stratified_split(df)
        if verbose:
            print("Datos: split canónico (UCI/caché local).")
        return df.loc[X_train.index].copy(), df.loc[X_test.index].copy()
    except Exception as exc:  # sin red ni caché → reconstrucción
        train_path = config.INPUTS_DIR / "arx_train.csv"
        test_path = config.INPUTS_DIR / "arx_test.csv"
        if not (train_path.exists() and test_path.exists()):
            raise RuntimeError(
                "Sin acceso a UCI/caché y sin arx_kit/inputs para reconstruir "
                f"el split (causa original: {exc})"
            ) from exc
        if verbose:
            print(f"Datos: split reconstruido desde {config.INPUTS_DIR} "
                  f"(UCI no disponible: {type(exc).__name__}).")

        def _read_auto_sep(path):
            # arx_train.csv usa ';' (formato ARX); arx_test.csv histórico
            # usa ','. Se detecta por la cabecera para tolerar ambos.
            with open(path, encoding="utf-8") as handle:
                header = handle.readline()
            return pd.read_csv(path, sep=";" if ";" in header else ",")

        return _read_auto_sep(train_path), _read_auto_sep(test_path)


def encoded_split_from_frames(df_train_raw: pd.DataFrame, df_test_raw: pd.DataFrame):
    """Reproduce las matrices de `stratified_split` desde los crudos: el OHE
    conjunto no depende del orden de filas, así que el column space es
    idéntico al canónico. Devuelve (X_train, X_test, y_train, y_test)."""
    from tfm.preprocessing import binarize_target

    y_train = binarize_target(df_train_raw[config.TARGET_COLUMN])
    y_test = binarize_target(df_test_raw[config.TARGET_COLUMN])

    n_train = len(df_train_raw)
    stacked = pd.concat(
        [df_train_raw.drop(columns=[config.TARGET_COLUMN]),
         df_test_raw.drop(columns=[config.TARGET_COLUMN])],
        ignore_index=True,
    )
    encoded = pd.get_dummies(stacked, drop_first=True)
    return encoded.iloc[:n_train], encoded.iloc[n_train:], y_train, y_test
