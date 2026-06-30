"""Fase 13: auditoría MIA reforzada. Tier 1 replica el Black-Box de la
Fase 6 (20 semillas, víctimas LR y RF) sobre los nueve escenarios de la
memoria; Tier 2 ejecuta LiRA offline (64 sombras, TPR a FPR bajo) sobre la
víctima LR en cuatro escenarios representativos."""

import re
import time
import unicodedata
from typing import Dict, List

import numpy as np
import pandas as pd

from tfm import config
from tfm.arx_io import load_arx_arrays
from tfm.data_loader import encoded_split_from_frames, load_split_frames
from tfm.mia import (
    aggregate_replicas,
    lira_metrics,
    lira_offline,
    run_mia_blackbox_replicas,
)
from tfm.models import (
    fit_lr_baseline,
    fit_lr_dp,
    fit_lr_dp_shadow,
    fit_lr_shadow,
    fit_rf_baseline,
)
from tfm.plotting import plot_lira_roc, plot_mia_replicas
from tfm.preprocessing import fit_scaler, percentile_data_norm

# Utilidad publicada del baseline LR (Fase 6): verificación de no-regresión
# de datos/entorno con independencia de la vía de carga.
PUBLISHED_BASELINE_UTILITY = 0.6178

# Mapeo de etiquetas de la tabla publicada (Fase 6) a las de esta fase.
PUBLISHED_LABEL_MAP = {
    "Baseline LR": "Baseline",
    "LR · k=10": "k=10",
    "LR · k=50": "k=50",
    "LR · DP ε=0.1": "DP ε=0.1",
    "LR · DP ε=1.0": "DP ε=1.0",
    "LR · DP ε=10.0": "DP ε=10.0",
}

VICTIM_LR = "Regresión Logística"
VICTIM_RF = "Random Forest"


def _slugify(label: str) -> str:
    """Convierte la etiqueta del escenario en un nombre de archivo portable."""
    text = label.replace("ε", "eps").replace("·", " ")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return text


class _Data:
    """Matrices canónicas compartidas por ambos tiers."""

    def __init__(self) -> None:
        df_train_raw, df_test_raw = load_split_frames()
        X_train_enc, X_test_enc, y_train, y_test = encoded_split_from_frames(
            df_train_raw, df_test_raw
        )
        scaler = fit_scaler(X_train_enc)
        self.X_train_scaled = scaler.transform(X_train_enc)
        self.X_test_scaled = scaler.transform(X_test_enc)
        self.y_train = y_train.values
        self.y_test = y_test.values
        self.data_norm = percentile_data_norm(self.X_train_scaled, percentile=95)

        # Para el OHE conjunto con los CSVs de ARX, los QID numéricos deben
        # ir como string (mismo formato que exportan las jerarquías).
        self.df_test_raw = df_test_raw.copy()
        for column in config.QID_NUMERIC_AS_STR:
            self.df_test_raw[column] = self.df_test_raw[column].astype(str)

        self._arx_cache: Dict[str, object] = {}

    def arx_arrays(self, filename: str):
        if filename not in self._arx_cache:
            filepath = config.ARX_OUTPUTS_DIR / filename
            self._arx_cache[filename] = load_arx_arrays(filepath, self.df_test_raw)
        return self._arx_cache[filename]


# Tier 1: réplicas del Black-Box. Escenarios en el orden de la tabla de la
# memoria, como (etiqueta, tipo, parámetro, víctimas); para DP solo LR.
TIER1_SCENARIO_DEFS = (
    [("Baseline", "baseline", None, (VICTIM_LR, VICTIM_RF))]
    + [(f"k={k}", "kanon", k, (VICTIM_LR, VICTIM_RF)) for k in (10, 50)]
    + [(f"DP ε={eps}", "dp", eps, (VICTIM_LR,)) for eps in (0.1, 1.0, 10.0)]
    + [(label, "lt", filename, (VICTIM_LR, VICTIM_RF))
       for label, filename in config.MIA_REPLICAS_LT_TARGETS]
)


def tier1_combos() -> List[tuple]:
    """Aplana TIER1_SCENARIO_DEFS en (etiqueta, tipo, parámetro, víctima)."""
    return [
        (label, kind, param, victim)
        for label, kind, param, victims in TIER1_SCENARIO_DEFS
        for victim in victims
    ]


def build_tier1_combo(data: _Data, label: str, kind: str, param, victim_name: str) -> dict:
    """Entrena la víctima de un combo y devuelve sus matrices de ataque;
    cada combo es autocontenido y determinista (RANDOM_STATE)."""
    if kind == "baseline":
        X_tr, y_tr, X_te = data.X_train_scaled, data.y_train, data.X_test_scaled
        fit = fit_lr_baseline if victim_name == VICTIM_LR else fit_rf_baseline
        model = fit(X_tr, y_tr)
    elif kind == "kanon" or kind == "lt":
        filename = config.arx_output_filename(param) if kind == "kanon" else param
        arrays = data.arx_arrays(filename)
        X_tr, y_tr, X_te = arrays.X_anon_scaled, arrays.y_anon, arrays.X_test_scaled
        fit = fit_lr_baseline if victim_name == VICTIM_LR else fit_rf_baseline
        model = fit(X_tr, y_tr)
    elif kind == "dp":
        X_tr, y_tr, X_te = data.X_train_scaled, data.y_train, data.X_test_scaled
        model = fit_lr_dp(X_tr, y_tr, float(param), data.data_norm)
    else:
        raise ValueError(f"Tipo de escenario Tier 1 desconocido: {kind}")

    return {
        "escenario": label, "victima": victim_name, "model": model,
        "X_tr": X_tr, "y_tr": y_tr, "X_te": X_te, "y_te": data.y_test,
    }


def replicate_combo(data: _Data, label: str, kind: str, param, victim_name: str) -> pd.DataFrame:
    """Entrena la víctima del combo y ejecuta las 20 réplicas (formato crudo)."""
    scenario = build_tier1_combo(data, label, kind, param, victim_name)
    df_raw = run_mia_blackbox_replicas(
        scenario["model"], scenario["X_tr"], scenario["y_tr"],
        scenario["X_te"], scenario["y_te"], config.DP_SEEDS[: config.MIA_N_REPLICAS],
    )
    df_raw.insert(0, "escenario", label)
    df_raw.insert(1, "victima", victim_name)
    return df_raw


def run_tier1(data: _Data) -> pd.DataFrame:
    print(f"\n--- Tier 1: réplicas Black-Box (n={config.MIA_N_REPLICAS} semillas) ---")

    raw_frames: List[pd.DataFrame] = []
    agg_rows: List[dict] = []
    for label, kind, param, victim_name in tier1_combos():
        start = time.time()
        df_raw = replicate_combo(data, label, kind, param, victim_name)
        raw_frames.append(df_raw)

        agg_rows.append({
            "escenario": label,
            "victima": victim_name,
            "utility_acc": float(df_raw["utility_acc"].iloc[0]),
            **aggregate_replicas(df_raw),
        })
        elapsed = time.time() - start
        print(
            f"  {label:<28} {victim_name:<20}"
            f" Adv={agg_rows[-1]['attack_advantage_mean']:+.4f}"
            f"±{agg_rows[-1]['attack_advantage_std']:.4f}"
            f"  AUC={agg_rows[-1]['attack_auc_mean']:.4f}"
            f"±{agg_rows[-1]['attack_auc_std']:.4f}  [{elapsed:.0f} s]"
        )

    df_raw_all = pd.concat(raw_frames, ignore_index=True)
    df_agg = pd.DataFrame(agg_rows)
    return finalize_tier1(df_raw_all, df_agg)


def finalize_tier1(df_raw_all: pd.DataFrame, df_agg: pd.DataFrame) -> pd.DataFrame:
    """Escribe CSVs y figura del Tier 1 y ejecuta la coherencia cruzada."""
    df_raw_all.to_csv(config.RESULTS_MIA_DIR / "mia_replicas_raw.csv", index=False)
    df_agg.to_csv(config.RESULTS_MIA_DIR / "mia_replicas.csv", index=False)

    df_published = _load_published_table()
    plot_mia_replicas(df_agg, config.RESULTS_DIR / "mia_replicas.png", df_published)
    if df_published is not None:
        _check_consistency(df_raw_all, df_published)
    return df_agg


def _load_published_table() -> pd.DataFrame:
    """Tabla de la Fase 6 con etiquetas traducidas a las de esta fase."""
    path = config.RESULTS_MIA_DIR / "mia_results.csv"
    if not path.exists():
        print("  Aviso: no se encuentra mia_results.csv; se omite la comparación.")
        return None
    df = pd.read_csv(path)
    df["escenario"] = df["escenario"].map(PUBLISHED_LABEL_MAP)
    return df.dropna(subset=["escenario"])


def _check_consistency(df_raw_all: pd.DataFrame, df_published: pd.DataFrame) -> None:
    """Coherencia cruzada: el valor publicado (una corrida) debe caer dentro
    de la banda [mín, máx] de las 20 réplicas de la víctima LR."""
    print("\n  Coherencia con la tabla publicada (Fase 6, víctima LR):")
    for _, pub in df_published.iterrows():
        replicas = df_raw_all[
            (df_raw_all["escenario"] == pub["escenario"])
            & (df_raw_all["victima"] == VICTIM_LR)
        ]
        for metric in ("attack_advantage", "attack_auc"):
            lo, hi = replicas[metric].min(), replicas[metric].max()
            ok = lo <= pub[metric] <= hi
            print(
                f"    {pub['escenario']:<12} {metric:<17}"
                f" publicado={pub[metric]:+.4f}  banda réplicas=[{lo:+.4f}, {hi:+.4f}]"
                f"  {'OK' if ok else 'FUERA DE BANDA'}"
            )


# Tier 2: LiRA offline sobre la víctima LR.

def run_lira_scenario(data: _Data, label: str, kind: str, param) -> tuple:
    """LiRA offline sobre un escenario, determinista (víctima RANDOM_STATE,
    objetivos DP_SEEDS[0], sombras base_seed). Devuelve (fila, fpr, tpr) y
    escribe el CSV de puntuaciones del escenario."""
    start = time.time()

    if kind == "baseline":
        victim = fit_lr_baseline(data.X_train_scaled, data.y_train)
        X_member_side, y_member_side = data.X_train_scaled, data.y_train
        X_test_side = data.X_test_scaled
        fit_fn = fit_lr_shadow
    elif kind == "kanon":
        arrays = data.arx_arrays(config.arx_output_filename(int(param)))
        victim = fit_lr_baseline(arrays.X_anon_scaled, arrays.y_anon)
        X_member_side, y_member_side = arrays.X_anon_scaled, arrays.y_anon
        X_test_side = arrays.X_test_scaled  # OHE conjunto del escenario
        fit_fn = fit_lr_shadow
    elif kind == "dp":
        victim = fit_lr_dp(data.X_train_scaled, data.y_train, float(param), data.data_norm)
        X_member_side, y_member_side = data.X_train_scaled, data.y_train
        X_test_side = data.X_test_scaled

        def fit_fn(X, y, seed, _eps=float(param), _norm=data.data_norm):
            return fit_lr_dp_shadow(X, y, _eps, _norm, seed)
    else:
        raise ValueError(f"Tipo de escenario LiRA desconocido: {kind}")

    # Población = lado member (train del escenario) ∪ lado non-member
    # (test en claro), en el espacio de features del escenario.
    X_pop = np.vstack([X_member_side, X_test_side])
    y_pop = np.concatenate([y_member_side, data.y_test])
    n_member_side = len(X_member_side)

    rng = np.random.default_rng(config.DP_SEEDS[0])
    n_targets = min(config.LIRA_N_TARGETS_PER_CLASS, n_member_side, len(X_test_side))
    member_idx = rng.choice(n_member_side, n_targets, replace=False)
    nonmember_idx = n_member_side + rng.choice(len(X_test_side), n_targets, replace=False)

    result = lira_offline(
        fit_fn, X_pop, y_pop, member_idx, nonmember_idx, victim,
        base_seed=config.RANDOM_STATE,
    )
    metrics = lira_metrics(result["scores"], result["labels"])

    pd.DataFrame({
        "is_member": result["labels"],
        "score": np.round(result["scores"], 6),
    }).to_csv(
        config.RESULTS_MIA_DIR / f"lira_scores_{_slugify(label)}.csv", index=False
    )

    elapsed = time.time() - start
    row = {
        "escenario": label,
        "n_shadows": config.MIA_SHADOW_COUNT,
        "n_targets_por_clase": n_targets,
        "victim_acc": result["victim_acc"],
        "shadow_acc_mean": result["shadow_acc_mean"],
        "shadow_acc_std": result["shadow_acc_std"],
        "auc": metrics["auc"],
        "advantage_max": metrics["advantage_max"],
        **{k: v for k, v in metrics.items() if k.startswith("tpr_at_fpr_")},
        "segundos": round(elapsed, 1),
    }
    print(
        f"  {label:<14} AUC={row['auc']:.4f}  "
        + "  ".join(
            f"TPR@{t:g}={row[f'tpr_at_fpr_{t:g}']:.4f}" for t in config.LIRA_FPR_TARGETS
        )
        + f"  acc sombras={row['shadow_acc_mean']:.4f}±{row['shadow_acc_std']:.4f}"
        f" (víctima {row['victim_acc']:.4f})  [{elapsed:.0f} s]"
    )
    return row, metrics["fpr"], metrics["tpr"]


def run_tier2(data: _Data) -> pd.DataFrame:
    print(f"\n--- Tier 2: LiRA offline ({config.MIA_SHADOW_COUNT} sombras, víctima LR) ---")
    rows: List[dict] = []
    roc_by_scenario: Dict[str, tuple] = {}
    for label, kind, param in config.LIRA_SCENARIOS:
        row, fpr, tpr = run_lira_scenario(data, label, kind, param)
        rows.append(row)
        roc_by_scenario[label] = (fpr, tpr)
    return finalize_tier2(pd.DataFrame(rows), roc_by_scenario)


def finalize_tier2(df_lira: pd.DataFrame, roc_by_scenario: Dict[str, tuple]) -> pd.DataFrame:
    """Escribe el CSV agregado y la figura ROC del Tier 2."""
    df_lira.to_csv(config.RESULTS_MIA_DIR / "mia_lira.csv", index=False)
    plot_lira_roc(roc_by_scenario, config.RESULTS_DIR / "lira_roc.png")
    return df_lira


def run() -> None:
    config.ensure_dirs()
    data = _Data()

    baseline_utility = None
    df_agg = run_tier1(data)
    baseline_lr = df_agg[(df_agg["escenario"] == "Baseline") & (df_agg["victima"] == VICTIM_LR)]
    if not baseline_lr.empty:
        baseline_utility = float(baseline_lr["utility_acc"].iloc[0])
        status = "OK" if abs(baseline_utility - PUBLISHED_BASELINE_UTILITY) < 5e-5 else "DIFIERE"
        print(
            f"\n  No-regresión de datos/entorno: utilidad baseline LR = "
            f"{baseline_utility:.4f} (publicado {PUBLISHED_BASELINE_UTILITY:.4f}) → {status}"
        )

    df_lira = run_tier2(data)

    print("\nResumen Tier 1 (agregado):")
    print(df_agg.drop(columns=[c for c in df_agg.columns if c.endswith("_ci95_lo") or c.endswith("_ci95_hi")])
          .round(4).to_string(index=False))
    print("\nResumen Tier 2 (LiRA):")
    print(df_lira.round(4).to_string(index=False))


if __name__ == "__main__":
    run()
