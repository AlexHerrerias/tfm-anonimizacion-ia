"""Auditoría adversarial mediante ataques de inferencia de membresía (MIA).

Coexisten dos implementaciones complementarias, ambas Black-Box con
clasificador atacante Random Forest (metodología de Shokri et al., 2017):

- `run_mia_blackbox` (Fase 6): recibe un modelo objetivo YA entrenado y
  devuelve métricas centradas en accuracy/advantage del atacante. Muestrea
  hasta 10k/5k ejemplos con un RNG sembrado por `config.RANDOM_STATE`.
- `mia_attack_rates` (Fase 9): entrena internamente la Regresión Logística
  objetivo y devuelve (TPR, FPR, Advantage, AUC). Usa un RNG independiente
  por configuración (`np.random.default_rng(seed)`), con la semilla tomada
  de `config.DP_SEEDS` por índice de escenario: esto elimina la dependencia
  del orden de las configuraciones y permite añadir o reordenar escenarios
  sin alterar los muestreos de los demás.

No se unifican en una sola función porque sus semánticas de muestreo y sus
métricas difieren y los números de la memoria dependen de cada una tal cual.
"""

from typing import Dict, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
from art.estimators.classification.scikitlearn import ScikitlearnLogisticRegression

from tfm import config


def run_mia_blackbox(
    sklearn_model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_train_sample: int = 10000,
    n_test_sample: int = 5000,
    seed: int = None,
) -> Dict[str, float]:
    """Ejecuta un MIA Black-Box sobre un modelo de Regresión Logística entrenado.

    Sigue la metodología de Shokri et al. (2017): el atacante construye un
    clasificador de pertenencia (Random Forest) sobre la mitad del conjunto
    auditor y se evalúa sobre la mitad restante.
    """
    seed = seed if seed is not None else config.RANDOM_STATE
    rng = np.random.default_rng(seed)

    n_tr = min(len(X_train), n_train_sample)
    n_te = min(len(X_test), n_test_sample)
    idx_tr = rng.choice(len(X_train), n_tr, replace=False)
    idx_te = rng.choice(len(X_test), n_te, replace=False)

    X_tr_sample = X_train[idx_tr]
    y_tr_sample = y_train[idx_tr]
    X_te_sample = X_test[idx_te]
    y_te_sample = y_test[idx_te]

    art_classifier = ScikitlearnLogisticRegression(model=sklearn_model)
    attack = MembershipInferenceBlackBox(estimator=art_classifier, attack_model_type="rf")

    half_tr = n_tr // 2
    half_te = n_te // 2

    attack.fit(
        x=X_tr_sample[:half_tr],
        y=y_tr_sample[:half_tr],
        test_x=X_te_sample[:half_te],
        test_y=y_te_sample[:half_te],
    )

    inferred_members = attack.infer(X_tr_sample[half_tr:], y_tr_sample[half_tr:])
    inferred_non_members = attack.infer(X_te_sample[half_te:], y_te_sample[half_te:])

    # Probabilidades del clasificador atacante para calcular AUC
    proba_members = attack.infer(X_tr_sample[half_tr:], y_tr_sample[half_tr:], probabilities=True)
    proba_non_members = attack.infer(X_te_sample[half_te:], y_te_sample[half_te:], probabilities=True)
    # ART devuelve probabilidad de la clase "member"; según versión puede ser (n,) o (n,1)/(n,2)
    proba_members = np.asarray(proba_members).reshape(len(proba_members), -1)
    proba_non_members = np.asarray(proba_non_members).reshape(len(proba_non_members), -1)
    # Tomar la última columna (prob. de "member" según convención ART)
    scores = np.concatenate([proba_members[:, -1], proba_non_members[:, -1]])
    labels = np.concatenate([
        np.ones(len(proba_members), dtype=int),
        np.zeros(len(proba_non_members), dtype=int),
    ])
    try:
        attack_auc = float(roc_auc_score(labels, scores))
    except ValueError:
        attack_auc = float("nan")

    n_correct = int((inferred_members == 1).sum()) + int((inferred_non_members == 0).sum())
    n_total = len(inferred_members) + len(inferred_non_members)
    attack_accuracy = n_correct / n_total

    rate_member = float((inferred_members == 1).mean())
    rate_non_member = float((inferred_non_members == 1).mean())
    advantage = rate_member - rate_non_member

    utility = float(accuracy_score(y_test, sklearn_model.predict(X_test)))

    return {
        "utility_acc": round(utility, 4),
        "attack_acc": round(attack_accuracy, 4),
        "attack_advantage": round(advantage, 4),
        "attack_auc": round(attack_auc, 4),
        "rate_member_predicted_1": round(rate_member, 4),
        "rate_non_member_predicted_1": round(rate_non_member, 4),
    }


def mia_attack_rates(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    n_attack: int = 2000,
) -> Tuple[float, float, float, float]:
    """Ejecuta el ataque MIA Black-Box con un RNG local sembrado por `seed`.

    Entrena internamente la Regresión Logística objetivo y devuelve
    (TPR, FPR, Advantage, AUC). Las matrices deben llegar en float32
    (ver `arx_io.load_arx_arrays(dtype=np.float32)`).
    """
    target = LogisticRegression(
        max_iter=1000,
        random_state=config.RANDOM_STATE,
        class_weight="balanced",
    )
    target.fit(X_train, y_train)
    art_classifier = ScikitlearnLogisticRegression(
        model=target, clip_values=(X_train.min(), X_train.max())
    )

    rng = np.random.default_rng(seed)
    n_attack = min(n_attack, len(X_test))
    attack_idx = rng.choice(len(X_train), n_attack, replace=False)
    test_idx = rng.choice(len(X_test), n_attack, replace=False)
    attack = MembershipInferenceBlackBox(
        estimator=art_classifier, attack_model_type="rf"
    )

    half = n_attack // 2
    attack.fit(
        X_train[attack_idx[:half]], y_train[attack_idx[:half]],
        X_test[test_idx[:half]], y_test[test_idx[:half]],
    )
    inf_members = attack.infer(X_train[attack_idx[half:]], y_train[attack_idx[half:]])
    inf_non_members = attack.infer(X_test[test_idx[half:]], y_test[test_idx[half:]])

    # AUC del atacante a partir de las probabilidades por ejemplo
    proba_m = np.asarray(
        attack.infer(X_train[attack_idx[half:]], y_train[attack_idx[half:]], probabilities=True)
    ).reshape(-1)
    proba_n = np.asarray(
        attack.infer(X_test[test_idx[half:]], y_test[test_idx[half:]], probabilities=True)
    ).reshape(-1)
    # Si vuelve (n,2) tras reshape, descartar primera mitad (clase 0)
    if len(proba_m) == 2 * half:
        proba_m = proba_m.reshape(half, 2)[:, -1]
        proba_n = proba_n.reshape(half, 2)[:, -1]
    scores = np.concatenate([proba_m, proba_n])
    labels = np.concatenate([np.ones(half, dtype=int), np.zeros(half, dtype=int)])
    try:
        auc = float(roc_auc_score(labels, scores))
    except ValueError:
        auc = float("nan")

    tpr = float(inf_members.mean())
    fpr = float(inf_non_members.mean())
    return tpr, fpr, tpr - fpr, auc
