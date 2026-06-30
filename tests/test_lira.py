"""Tests unitarios del scoring LiRA y de las réplicas MIA (Fase 13).

cada test valida una propiedad estadística que debe cumplirse por construcción,
de modo que si falla algo es por que esta mal desarrollado.
"""

import sys
import warnings

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from tfm.mia import (
    lira_metrics,
    lira_offline,
    run_mia_blackbox_replicas,
    scaled_logit_confidences,
)

warnings.filterwarnings("ignore")

SEED = 42


def _fit_lr(X, y, seed):
    return LogisticRegression(max_iter=500, random_state=seed).fit(X, y)


def _fit_tree(X, y, seed):
    # Árbol sin podar: memoriza su conjunto de entrenamiento.
    return DecisionTreeClassifier(max_depth=None, random_state=seed).fit(X, y)


def test_h0_calibration():
    """Víctima que no vio los objetivos ⇒ TPR ≈ FPR (el ataque no inventa fuga)."""
    X, y = make_classification(
        n_samples=6000, n_features=12, n_informative=6, random_state=SEED
    )
    X_pop, y_pop = X[:4000], y[:4000]          # población de sombras y objetivos
    victim = _fit_lr(X[4000:], y[4000:], SEED)  # entrenada FUERA de la población

    member_idx = np.arange(0, 500)              # "members" ficticios: la víctima
    nonmember_idx = np.arange(3500, 4000)       # tampoco los vio

    result = lira_offline(
        _fit_lr, X_pop, y_pop, member_idx, nonmember_idx, victim,
        n_shadows=32, shadow_fraction=0.5, base_seed=SEED,
    )
    metrics = lira_metrics(result["scores"], result["labels"], fpr_targets=[0.01, 0.1])

    assert 0.43 <= metrics["auc"] <= 0.57, f"H0: AUC={metrics['auc']} lejos de 0.5"
    assert 0.03 <= metrics["tpr_at_fpr_0.1"] <= 0.20, (
        f"H0: TPR@10%FPR={metrics['tpr_at_fpr_0.1']} lejos de la diagonal"
    )
    print(f"  H0: AUC={metrics['auc']:.3f}, TPR@10%FPR={metrics['tpr_at_fpr_0.1']:.3f} ≈ diagonal  ✓")


def test_positive_detection():
    """Víctima sobreajustada ⇒ el ataque debe detectar fuga a FPR bajo."""
    # flip_y alto: etiquetas ruidosas que solo un modelo memorizador "acierta"
    X, y = make_classification(
        n_samples=4000, n_features=12, n_informative=5, flip_y=0.3, random_state=SEED
    )
    train_idx = np.arange(0, 2000)
    victim = _fit_tree(X[train_idx], y[train_idx], SEED)

    rng = np.random.default_rng(SEED)
    member_idx = rng.choice(2000, 400, replace=False)
    nonmember_idx = 2000 + rng.choice(2000, 400, replace=False)

    result = lira_offline(
        _fit_tree, X, y, member_idx, nonmember_idx, victim,
        n_shadows=32, shadow_fraction=0.5, base_seed=SEED,
    )
    metrics = lira_metrics(result["scores"], result["labels"], fpr_targets=[0.01, 0.1])

    assert metrics["auc"] > 0.65, f"Detección: AUC={metrics['auc']} demasiado bajo"
    assert metrics["tpr_at_fpr_0.01"] > 0.05, (
        f"Detección: TPR@1%FPR={metrics['tpr_at_fpr_0.01']} no supera claramente 0.01"
    )
    print(
        f"  Detección: AUC={metrics['auc']:.3f}, "
        f"TPR@1%FPR={metrics['tpr_at_fpr_0.01']:.3f} ≫ 0.01  ✓"
    )


def test_determinism():
    """Mismas semillas ⇒ resultados idénticos (réplicas y LiRA)."""
    X, y = make_classification(n_samples=2000, n_features=8, random_state=SEED)
    Xtr, Xte, ytr, yte = X[:1400], X[1400:], y[:1400], y[1400:]
    victim = _fit_lr(Xtr, ytr, SEED)

    df_a = run_mia_blackbox_replicas(victim, Xtr, ytr, Xte, yte, seeds=[42, 137, 271])
    df_b = run_mia_blackbox_replicas(victim, Xtr, ytr, Xte, yte, seeds=[42, 137, 271])
    assert df_a.equals(df_b), "Réplicas no deterministas con las mismas semillas"

    member_idx = np.arange(0, 200)
    nonmember_idx = np.arange(1400, 1600)
    s_a = lira_offline(_fit_lr, X, y, member_idx, nonmember_idx, victim,
                       n_shadows=8, shadow_fraction=0.5, base_seed=SEED)["scores"]
    s_b = lira_offline(_fit_lr, X, y, member_idx, nonmember_idx, victim,
                       n_shadows=8, shadow_fraction=0.5, base_seed=SEED)["scores"]
    assert np.array_equal(s_a, s_b), "LiRA no determinista con la misma base_seed"
    print("  Determinismo: réplicas y LiRA reproducen byte a byte  ✓")


def test_tpr_at_fpr_metric():
    """Casos degenerados con ROC conocida."""
    # Separación perfecta: TPR=1 a cualquier FPR
    scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    labels = np.array([1, 1, 1, 0, 0, 0])
    metrics = lira_metrics(scores, labels, fpr_targets=[0.001, 0.1])
    assert metrics["auc"] == 1.0
    assert metrics["tpr_at_fpr_0.001"] == 1.0
    assert metrics["advantage_max"] == 1.0

    # Scores constantes: sin información ⇒ a FPR<1 no se alcanza TPR>0
    # (la ROC solo tiene los puntos (0,0) y (1,1))
    scores = np.full(100, 0.5)
    labels = np.array([1, 0] * 50)
    metrics = lira_metrics(scores, labels, fpr_targets=[0.1])
    assert metrics["tpr_at_fpr_0.1"] == 0.0
    assert metrics["advantage_max"] == 0.0
    print("  Métrica TPR@FPR: casos degenerados correctos  ✓")


def test_scaled_logit():
    """El scaled logit debe ser monótono en p y estar acotado por el clip."""
    X, y = make_classification(n_samples=300, n_features=6, random_state=SEED)
    model = _fit_lr(X, y, SEED)
    phi = scaled_logit_confidences(model, X, y)
    assert np.all(np.isfinite(phi)), "φ no finito pese al recorte de p"
    # Cota del clip: |φ| ≤ logit(1−1e−7) ≈ 16.12
    assert np.all(np.abs(phi) <= 16.2), "φ fuera de la cota del recorte"
    print("  Scaled logit: finito y acotado por el recorte  ✓")


ALL_TESTS = [
    test_scaled_logit,
    test_tpr_at_fpr_metric,
    test_determinism,
    test_h0_calibration,
    test_positive_detection,
]


if __name__ == "__main__":
    print("Tests del scoring LiRA / réplicas MIA (Fase 13):")
    failures = 0
    for test in ALL_TESTS:
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"  FALLO en {test.__name__}: {exc}")
    if failures:
        sys.exit(f"{failures} test(s) fallidos")
    print(f"\n{len(ALL_TESTS)} tests superados.")
