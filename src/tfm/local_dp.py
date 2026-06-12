"""Privacidad Diferencial Local (LDP): perturbación en origen del conjunto de
entrenamiento.

Cierra el eje experimental que la memoria describe en la Subsección
«Paradigmas de aplicación: DP Local y DP Global»: frente a la DP global
(diffprivlib, ruido durante el entrenamiento por un curador de confianza),
aquí el ruido se inyecta REGISTRO A REGISTRO antes de que el dato llegue al
agregador, que se asume no confiable.

Modelo de amenaza y decisiones de diseño
----------------------------------------
* Cada titular del registro aleatoriza localmente sus 45 atributos
  predictores; el agregador solo observa registros perturbados.
* La etiqueta `readmitted` NO se perturba: en el escenario clínico la
  readmisión la observa el propio sistema central (el paradigma con etiqueta
  privada es un setting aparte, *label-DP*, y mezclarlo impediría atribuir
  la pérdida de utilidad a la perturbación de los atributos).
* El conjunto de test permanece en claro: el modelo se entrena sobre datos
  protegidos y en producción puntúa registros reales, simétrico al diseño
  de las fases ARX y DP global.
* Presupuesto user-level con composición secuencial (Dwork): ε_j = ε / n_eff,
  con n_eff = nº de atributos perturbables (se excluyen las columnas
  constantes: su dominio es unitario, el valor es conocimiento público y
  el mecanismo identidad satisface ε = 0).

Mecanismos
----------
* Categóricas (y códigos administrativos nominales): k-RR / *Generalized
  Randomized Response* (Warner, 1965; Kairouz et al., 2016): se reporta el
  valor verdadero con probabilidad e^ε/(e^ε + d - 1) y un valor uniforme
  del resto del dominio en caso contrario.
* Numéricas (recuentos clínicos acotados): mecanismo de Laplace con
  sensibilidad igual al ancho del rango público del atributo, con recorte
  posterior al rango y redondeo a entero (ambos post-procesado, no
  consumen presupuesto).

Los dominios y rangos se toman del conjunto publicado (metadatos públicos
del dataset UCI 296), nunca del registro individual.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from tfm import config
from tfm.models import build_baseline_models
from tfm.preprocessing import binarize_target, joint_ohe

# Modelos entrenados en el barrido LDP. La privacidad reside en los DATOS,
# no en el algoritmo, por lo que se usan los estimadores estándar de
# scikit-learn con los hiperparámetros del baseline; se restringe a los dos
# modelos del barrido de DP global para mantener la comparativa simétrica.
LDP_MODEL_NAMES = ("Regresión Logística", "Naive Bayes (Gaussian)")


@dataclass(frozen=True)
class AttributeSpec:
    """Especificación de perturbación de un atributo.

    kind:
      * "categorical": k-RR sobre `domain`.
      * "numeric":     Laplace sobre `bounds` (rango público), redondeo a entero.
      * "constant":    dominio unitario; no se perturba ni consume presupuesto.
    """

    name: str
    kind: str
    domain: Optional[Tuple] = None
    bounds: Optional[Tuple[float, float]] = None


def build_attribute_specs(df_features: pd.DataFrame) -> List[AttributeSpec]:
    """Deriva la especificación de mecanismo por atributo.

    El dominio/rango se calcula sobre el conjunto completo publicado
    (metadato público), no sobre el registro individual. Las columnas de
    `config.LDP_NOMINAL_INT_COLUMNS` son enteras pero nominales y reciben
    k-RR; el resto de numéricas se tratan como recuentos acotados.
    """
    specs: List[AttributeSpec] = []
    for column in df_features.columns:
        series = df_features[column]
        unique_values = pd.unique(series.dropna())
        if len(unique_values) <= 1:
            specs.append(AttributeSpec(name=column, kind="constant"))
        elif series.dtype == object or column in config.LDP_NOMINAL_INT_COLUMNS:
            specs.append(
                AttributeSpec(name=column, kind="categorical", domain=tuple(sorted(unique_values)))
            )
        else:
            specs.append(
                AttributeSpec(
                    name=column,
                    kind="numeric",
                    bounds=(float(series.min()), float(series.max())),
                )
            )
    return specs


def effective_attribute_count(specs: List[AttributeSpec]) -> int:
    """Número de atributos que consumen presupuesto (excluye constantes)."""
    return sum(1 for spec in specs if spec.kind != "constant")


def krr_perturb(values: np.ndarray, domain: Tuple, epsilon: float, rng: np.random.Generator) -> np.ndarray:
    """k-RR vectorizado: conserva el valor con p = e^ε/(e^ε + d - 1).

    Cuando no conserva, sustituye por un valor uniforme del RESTO del
    dominio (formulación estándar de Kairouz et al., 2016).
    """
    domain_array = np.asarray(domain, dtype=object)
    d = len(domain_array)
    p_keep = np.exp(epsilon) / (np.exp(epsilon) + d - 1)

    keep_mask = rng.random(len(values)) < p_keep
    result = np.array(values, dtype=object, copy=True)

    n_replace = int((~keep_mask).sum())
    if n_replace == 0:
        return result

    # Uniforme sobre el dominio excluyendo el valor verdadero: se muestrea un
    # desplazamiento 1..d-1 sobre la posición del valor real en el dominio.
    value_to_index = {value: index for index, value in enumerate(domain_array)}
    true_indices = np.array([value_to_index[v] for v in result[~keep_mask]])
    offsets = rng.integers(1, d, size=n_replace)
    result[~keep_mask] = domain_array[(true_indices + offsets) % d]
    return result


def laplace_perturb(
    values: np.ndarray,
    bounds: Tuple[float, float],
    epsilon: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Mecanismo de Laplace local sobre un atributo numérico acotado.

    Sensibilidad = ancho del rango público (un registro puede cambiar el
    valor reportado en todo el rango). El recorte al rango y el redondeo a
    entero son post-procesado y no consumen presupuesto adicional.
    """
    low, high = bounds
    scale = (high - low) / epsilon
    noisy = values.astype(float) + rng.laplace(loc=0.0, scale=scale, size=len(values))
    return np.clip(np.round(noisy), low, high).astype(int)


def randomize_dataframe(
    df_features: pd.DataFrame,
    epsilon_total: float,
    specs: List[AttributeSpec],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Aplica LDP registro a registro con presupuesto user-level ε_total.

    El presupuesto se reparte uniformemente entre los atributos perturbables
    (composición secuencial): ε_j = ε_total / n_eff.
    """
    n_eff = effective_attribute_count(specs)
    epsilon_per_attribute = epsilon_total / n_eff

    df_perturbed = df_features.copy()
    for spec in specs:
        if spec.kind == "constant":
            continue
        values = df_perturbed[spec.name].values
        if spec.kind == "categorical":
            perturbed = krr_perturb(values, spec.domain, epsilon_per_attribute, rng)
        else:
            perturbed = laplace_perturb(values, spec.bounds, epsilon_per_attribute, rng)
        # Conservar el dtype original: los códigos nominales enteros deben
        # seguir siendo numéricos para que el column space del OHE coincida
        # con el del baseline (k-RR devuelve object).
        df_perturbed[spec.name] = np.asarray(perturbed).astype(df_features[spec.name].dtype)
    return df_perturbed


def _fit_eval_models(
    X_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    races_test: np.ndarray,
) -> Tuple[List[Dict], List[Dict]]:
    """Entrena LR/NB sobre el train perturbado y evalúa sobre test en claro.

    Devuelve (filas_utilidad, filas_fairness); el desglose por subgrupo
    `race` se calcula en la misma pasada (solo LR, igual que en DP global)
    para no duplicar entrenamientos.
    """
    models = {name: model for name, model in build_baseline_models().items() if name in LDP_MODEL_NAMES}
    utility_rows: List[Dict] = []
    fairness_rows: List[Dict] = []
    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        utility_rows.append({
            "modelo": model_name,
            "Accuracy": accuracy_score(y_test, predictions),
            "F1-Score": f1_score(y_test, predictions, average="weighted"),
        })
        if model_name == "Regresión Logística":
            for race_value in np.unique(races_test):
                mask = races_test == race_value
                fairness_rows.append({
                    "race": race_value,
                    "n": int(mask.sum()),
                    "Accuracy": float(accuracy_score(y_test[mask], predictions[mask])),
                })
    return utility_rows, fairness_rows


def sweep_ldp(
    df_train_raw: pd.DataFrame,
    df_test_raw: pd.DataFrame,
    epsilons: List[float] = None,
    n_repetitions: int = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Barrido LDP completo: ε × repeticiones × 2 modelos.

    En cada repetición se re-aleatoriza el conjunto de entrenamiento (la
    estocasticidad de la LDP reside en los datos, no en el estimador) con la
    semilla dispersa correspondiente de `config.DP_SEEDS`, y se reconstruye
    el pipeline OHE conjunto + estandarización exactamente como en la fase
    ARX. Devuelve (df_utilidad, df_fairness) con una fila por celda.
    """
    epsilons = epsilons or config.LDP_EPSILON_VALUES
    n_repetitions = n_repetitions or config.N_REPETITIONS_DP
    seeds = config.DP_SEEDS[:n_repetitions]

    X_train_raw = df_train_raw.drop(columns=[config.TARGET_COLUMN])
    X_test_raw = df_test_raw.drop(columns=[config.TARGET_COLUMN])
    y_train = binarize_target(df_train_raw[config.TARGET_COLUMN]).values
    y_test = binarize_target(df_test_raw[config.TARGET_COLUMN]).values
    races_test = df_test_raw["race"].values

    specs = build_attribute_specs(X_train_raw)
    n_eff = effective_attribute_count(specs)
    print(
        f"LDP: {len(specs)} atributos, {n_eff} perturbables "
        f"({sum(1 for s in specs if s.kind == 'categorical')} categóricos k-RR, "
        f"{sum(1 for s in specs if s.kind == 'numeric')} numéricos Laplace, "
        f"{sum(1 for s in specs if s.kind == 'constant')} constantes exentos)"
    )

    utility_rows: List[Dict] = []
    fairness_rows: List[Dict] = []
    for epsilon in epsilons:
        for rep, seed in enumerate(seeds):
            rng = np.random.default_rng(seed)
            X_train_perturbed = randomize_dataframe(X_train_raw, epsilon, specs, rng)

            X_train_array, X_test_array = joint_ohe(X_train_perturbed, X_test_raw)
            scaler = StandardScaler().fit(X_train_array)
            X_train_scaled = scaler.transform(X_train_array)
            X_test_scaled = scaler.transform(X_test_array)

            cell_utility, cell_fairness = _fit_eval_models(
                X_train_scaled, X_test_scaled, y_train, y_test, races_test
            )
            for row in cell_utility:
                utility_rows.append({
                    "epsilon": epsilon, "rep": rep, "seed": seed,
                    "epsilon_por_atributo": epsilon / n_eff, **row,
                })
            for row in cell_fairness:
                fairness_rows.append({"epsilon": epsilon, "rep": rep, **row})
        print(f"  ε = {epsilon} completado ({n_repetitions} réplicas)")

    return pd.DataFrame(utility_rows), pd.DataFrame(fairness_rows)
