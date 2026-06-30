"""Ataques de inferencia de membresía: Black-Box de ART (fases 6 y 9, con
semánticas de muestreo propias que se conservan tal cual), réplicas con
semillas dispersas y LiRA offline con modelos sombra (fase 13)."""

from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import ndtr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
from art.estimators.classification import SklearnClassifier
from art.estimators.classification.scikitlearn import ScikitlearnLogisticRegression

from tfm import config


def run_mia_blackbox(
    sklearn_model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = None,
) -> Dict[str, float]:
    """MIA Black-Box (Shokri et al., 2017): clasificador de pertenencia RF
    entrenado sobre la mitad del conjunto auditor (10k members / 5k
    non-members) y evaluado sobre el resto."""
    seed = seed if seed is not None else config.RANDOM_STATE
    rng = np.random.default_rng(seed)

    n_tr = min(len(X_train), 10000)
    n_te = min(len(X_test), 5000)
    idx_tr = rng.choice(len(X_train), n_tr, replace=False)
    idx_te = rng.choice(len(X_test), n_te, replace=False)

    X_tr_sample = X_train[idx_tr]
    y_tr_sample = y_train[idx_tr]
    X_te_sample = X_test[idx_te]
    y_te_sample = y_test[idx_te]

    # Wrapper explícito para LR (el factory de ART rechaza la subclase de
    # diffprivlib); para el resto de víctimas el factory elige el adecuado.
    if isinstance(sklearn_model, LogisticRegression):
        art_classifier = ScikitlearnLogisticRegression(model=sklearn_model)
    else:
        art_classifier = SklearnClassifier(model=sklearn_model)
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
    """Variante de la Fase 9: entrena internamente la LR objetivo y devuelve
    (TPR, FPR, Advantage, AUC); las matrices deben llegar en float32."""
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


# Fase 13, Tier 1: réplicas del ataque Black-Box.

def run_mia_blackbox_replicas(
    sklearn_model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seeds: Iterable[int],
) -> pd.DataFrame:
    """Replica `run_mia_blackbox` por semilla: cada una fija el muestreo del
    auditor y el RNG global de numpy (del que depende el RF atacante de ART),
    haciendo cada réplica determinista. Devuelve una fila por semilla."""
    rows: List[dict] = []
    for seed in seeds:
        np.random.seed(seed % (2**32))  # determinismo del atacante RF de ART
        rows.append({
            "seed": seed,
            **run_mia_blackbox(sklearn_model, X_train, y_train, X_test, y_test, seed=seed),
        })
    return pd.DataFrame(rows)


def aggregate_replicas(
    df_raw: pd.DataFrame,
    metrics: Tuple[str, ...] = ("attack_advantage", "attack_auc", "attack_acc"),
) -> Dict[str, float]:
    """Agrega réplicas: media, desviación típica e IC 95 % con t de Student
    (n-1 grados de libertad)."""
    out: Dict[str, float] = {"n_replicas": int(len(df_raw))}
    n = len(df_raw)
    t_crit = stats.t.ppf(0.975, n - 1) if n > 1 else float("nan")
    for metric in metrics:
        values = df_raw[metric].to_numpy(dtype=float)
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if n > 1 else 0.0
        half = float(t_crit * std / np.sqrt(n)) if n > 1 else 0.0
        out[f"{metric}_mean"] = round(mean, 4)
        out[f"{metric}_std"] = round(std, 4)
        out[f"{metric}_ci95_lo"] = round(mean - half, 4)
        out[f"{metric}_ci95_hi"] = round(mean + half, 4)
    return out


# Fase 13, Tier 2: LiRA offline (Carlini et al., 2022).

def scaled_logit_confidences(model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Estadístico LiRA por ejemplo: phi = log(p/(1-p)), con p la confianza
    en la clase verdadera recortada a [1e-7, 1-1e-7]."""
    proba = model.predict_proba(X)
    class_index = np.searchsorted(model.classes_, y)
    p = proba[np.arange(len(y)), class_index]
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    return np.log(p) - np.log1p(-p)


def lira_offline(
    fit_shadow_fn: Callable[[np.ndarray, np.ndarray, int], object],
    X_pop: np.ndarray,
    y_pop: np.ndarray,
    member_idx: np.ndarray,
    nonmember_idx: np.ndarray,
    victim_model,
    n_shadows: int = config.MIA_SHADOW_COUNT,
    shadow_fraction: float = config.LIRA_SHADOW_FRACTION,
    base_seed: int = config.RANDOM_STATE,
    min_out_shadows: int = config.LIRA_MIN_OUT_SHADOWS,
) -> Dict[str, object]:
    """LiRA offline: sombras sobre subconjuntos aleatorios de la población,
    gaussiana N(mu_out, sigma_out) por objetivo con los phi de las sombras
    que no lo vieron, y score Phi(z), creciente con la evidencia de member."""
    rng = np.random.default_rng(base_seed)
    n_pop = len(X_pop)
    n_sub = int(round(shadow_fraction * n_pop))

    target_idx = np.concatenate([member_idx, nonmember_idx])
    labels = np.concatenate([
        np.ones(len(member_idx), dtype=int),
        np.zeros(len(nonmember_idx), dtype=int),
    ])
    X_t = X_pop[target_idx]
    y_t = y_pop[target_idx]
    nonmember_slice = slice(len(member_idx), len(target_idx))

    phi = np.empty((n_shadows, len(target_idx)), dtype=np.float32)
    in_shadow = np.zeros((n_shadows, len(target_idx)), dtype=bool)
    shadow_accs: List[float] = []

    for s in range(n_shadows):
        shadow_seed = int(rng.integers(0, 2**31 - 1))
        subset = rng.choice(n_pop, n_sub, replace=False)
        shadow = fit_shadow_fn(X_pop[subset], y_pop[subset], shadow_seed)

        phi[s] = scaled_logit_confidences(shadow, X_t, y_t)
        member_mask = np.zeros(n_pop, dtype=bool)
        member_mask[subset] = True
        in_shadow[s] = member_mask[target_idx]
        shadow_accs.append(
            float(accuracy_score(y_t[nonmember_slice], shadow.predict(X_t[nonmember_slice])))
        )

    out_mask = ~in_shadow
    out_counts = out_mask.sum(axis=0)
    phi_out_sum = np.where(out_mask, phi, 0.0).sum(axis=0)
    mu_out = phi_out_sum / np.maximum(out_counts, 1)
    var_out = np.where(out_mask, (phi - mu_out) ** 2, 0.0).sum(axis=0) / np.maximum(out_counts - 1, 1)
    sigma_out = np.sqrt(var_out)

    # Suavizado y fallback global para ejemplos con pocas sombras OUT
    global_sigma = float(np.median(sigma_out[out_counts >= min_out_shadows]))
    sigma_out = np.where(out_counts < min_out_shadows, global_sigma, sigma_out)
    sigma_out = np.maximum(sigma_out, 1e-3)

    phi_victim = scaled_logit_confidences(victim_model, X_t, y_t)
    z = (phi_victim - mu_out) / sigma_out
    scores = ndtr(z)  # Φ(z)

    victim_acc = float(accuracy_score(y_t[nonmember_slice], victim_model.predict(X_t[nonmember_slice])))

    return {
        "scores": scores,
        "labels": labels,
        "shadow_acc_mean": round(float(np.mean(shadow_accs)), 4),
        "shadow_acc_std": round(float(np.std(shadow_accs, ddof=1)), 4),
        "victim_acc": round(victim_acc, 4),
        "min_out_counts": int(out_counts.min()),
    }


def lira_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    fpr_targets: Iterable[float] = None,
) -> Dict[str, object]:
    """Métricas LiRA: ROC completa, AUC, Advantage máxima y TPR a cada FPR
    objetivo (interpolación escalonada, convención de Carlini et al.)."""
    fpr_targets = list(fpr_targets) if fpr_targets is not None else list(config.LIRA_FPR_TARGETS)
    fpr, tpr, _ = roc_curve(labels, scores)
    auc = float(roc_auc_score(labels, scores))
    advantage_max = float(np.max(tpr - fpr))

    out: Dict[str, object] = {
        "auc": round(auc, 4),
        "advantage_max": round(advantage_max, 4),
        "fpr": fpr,
        "tpr": tpr,
    }
    for target in fpr_targets:
        pos = np.searchsorted(fpr, target, side="right") - 1
        out[f"tpr_at_fpr_{target:g}"] = round(float(tpr[max(pos, 0)]), 5)
    return out
