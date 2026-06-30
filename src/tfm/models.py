"""Constructores de modelos baseline y diferencialmente privados."""

from typing import Dict, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

import diffprivlib.models as dpm

from tfm import config


def build_baseline_models() -> Dict[str, object]:
    """Cuatro modelos de referencia con hiperparámetros documentados en el TFM."""
    return {
        "Regresión Logística": LogisticRegression(
            max_iter=1000,
            random_state=config.RANDOM_STATE,
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=config.RANDOM_STATE,
            class_weight="balanced",
        ),
        "Árbol de Decisión": DecisionTreeClassifier(
            max_depth=10,
            random_state=config.RANDOM_STATE,
            class_weight="balanced",
        ),
        "Naive Bayes (Gaussian)": GaussianNB(),
    }


def build_dp_models(
    epsilon: float,
    data_norm: float,
    bounds: Tuple[np.ndarray, np.ndarray],
    random_state: int,
) -> Dict[str, object]:
    """Versiones diferencialmente privadas de los dos modelos comparables;
    max_iter=100 evita la no-convergencia de L-BFGS con gradientes ruidosos."""
    return {
        "Regresión Logística": dpm.LogisticRegression(
            epsilon=epsilon,
            data_norm=data_norm,
            max_iter=100,
            random_state=random_state,
        ),
        "Naive Bayes (Gaussian)": dpm.GaussianNB(
            epsilon=epsilon,
            bounds=bounds,
            random_state=random_state,
        ),
    }


def fit_lr_baseline(X_train_scaled, y_train) -> LogisticRegression:
    """Regresión Logística de referencia entrenada (modelo objetivo de la MIA)."""
    model = LogisticRegression(
        max_iter=1000,
        random_state=config.RANDOM_STATE,
        class_weight="balanced",
    )
    model.fit(X_train_scaled, y_train)
    return model


def fit_lr_dp(X_train_scaled, y_train, epsilon: float, data_norm: float):
    """Regresión Logística diferencialmente privada entrenada (objetivo MIA)."""
    model = dpm.LogisticRegression(
        epsilon=epsilon,
        data_norm=data_norm,
        max_iter=100,
        random_state=config.RANDOM_STATE,
    )
    model.fit(X_train_scaled, y_train)
    return model


def fit_rf_baseline(X_train_scaled, y_train) -> RandomForestClassifier:
    """Random Forest entrenado con los hiperparámetros del baseline
    (segunda víctima MIA de la Fase 13, de mayor capacidad que la LR)."""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=config.RANDOM_STATE,
        class_weight="balanced",
    )
    model.fit(X_train_scaled, y_train)
    return model


def fit_lr_shadow(X, y, random_state: int) -> LogisticRegression:
    """Sombra LiRA: idéntica a `fit_lr_baseline` salvo la semilla,
    que varía por sombra."""
    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        class_weight="balanced",
    )
    model.fit(X, y)
    return model


def fit_lr_dp_shadow(X, y, epsilon: float, data_norm: float, random_state: int):
    """Sombra LiRA del escenario DP: la semilla por sombra materializa la
    aleatoriedad del mecanismo que el atacante debe calibrar."""
    model = dpm.LogisticRegression(
        epsilon=epsilon,
        data_norm=data_norm,
        max_iter=100,
        random_state=random_state,
    )
    model.fit(X, y)
    return model


def predict_baselines(X_train_scaled, X_test_scaled, y_train) -> Dict[str, np.ndarray]:
    """Entrena los cuatro baselines y devuelve sus predicciones sobre test.
    Las matrices deben venir de `stratified_split + fit_scaler` para
    reproducir byte a byte el baseline de la Fase 1."""
    predictions: Dict[str, np.ndarray] = {}
    for name, model in build_baseline_models().items():
        model.fit(X_train_scaled, y_train)
        predictions[name] = model.predict(X_test_scaled)
    return predictions
