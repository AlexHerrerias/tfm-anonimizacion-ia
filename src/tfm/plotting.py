"""Generación de figuras consumidas por la memoria del TFM."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tfm import config


def plot_kanon_degradation(df_results: pd.DataFrame, output_path: Path) -> None:
    """Curva de degradación de Accuracy y F1 a lo largo del barrido de k.

    Disposición vertical (Accuracy arriba, F1 abajo), consistente con los
    pies de figura de la memoria.
    """
    fig, axes = plt.subplots(2, 1, figsize=(9, 10))
    k_values_full = [0] + list(config.K_VALUES)

    for model in df_results["Modelo"].unique():
        subset = df_results[df_results["Modelo"] == model].sort_values("k")
        axes[0].plot(subset["k"], subset["Accuracy"], marker="o", label=model)
        axes[1].plot(subset["k"], subset["F1-Score"], marker="s", label=model)

    for ax, ylabel, title in [
        (axes[0], "Accuracy", "Degradación de Accuracy con k"),
        (axes[1], "F1-Score", "Degradación de F1-Score con k"),
    ]:
        ax.set_xlabel("k (privacidad sintáctica)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xscale("symlog")
        ax.set_xticks(k_values_full)
        ax.set_xticklabels([str(k) for k in k_values_full])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_dp_degradation(df_results_aggregated: pd.DataFrame, baselines: dict, output_path: Path) -> None:
    """Curva de degradación de Accuracy/F1 vs ε con bandas de error.

    Disposición vertical (Accuracy arriba, F1 abajo), consistente con los
    pies de figura de la memoria.
    """
    fig, axes = plt.subplots(2, 1, figsize=(9, 10))
    epsilons = sorted(df_results_aggregated.index.get_level_values("epsilon").unique())

    for ax, mean_col, std_col, ylabel, metric_key in [
        (axes[0], "Acc_mean", "Acc_std", "Accuracy", "acc"),
        (axes[1], "F1_mean", "F1_std", "F1-Score (ponderado)", "f1"),
    ]:
        for model in ("Regresión Logística", "Naive Bayes (Gaussian)"):
            means = [df_results_aggregated.loc[(model, e), mean_col] for e in epsilons]
            stds = [df_results_aggregated.loc[(model, e), std_col] for e in epsilons]
            ax.errorbar(epsilons, means, yerr=stds, marker="o", capsize=4, label=model)

        for color, model_key in zip(("C0", "C1"), ("Regresión Logística", "Naive Bayes (Gaussian)")):
            ax.axhline(
                baselines[model_key][0 if metric_key == "acc" else 1],
                ls="--",
                color=color,
                alpha=0.5,
                lw=1,
                label=f"Baseline {model_key.split()[0]}",
            )

        ax.set_xscale("log")
        ax.set_xticks(epsilons)
        ax.set_xticklabels([str(e) for e in epsilons])
        ax.set_xlabel(r"Presupuesto de privacidad $\epsilon$")
        ax.set_ylabel(ylabel)
        ax.set_title(rf"Privacidad Diferencial: {ylabel} vs $\epsilon$ (mean$\pm$std, $n={config.N_REPETITIONS_DP}$)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_fairness_curves(
    df_fairness: pd.DataFrame,
    x_column: str,
    x_label: str,
    output_path: Path,
    title: str,
) -> None:
    """Curvas por subgrupo `race` para fairness ARX o DP."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    pivot = df_fairness.pivot_table(index="race", columns=x_column, values="Accuracy", aggfunc="mean")

    for race_value in pivot.index:
        n_subgroup = int(df_fairness[df_fairness["race"] == race_value]["n"].iloc[0])
        ax.plot(pivot.columns.tolist(), pivot.loc[race_value].values, marker="o",
                label=f"{race_value} (n={n_subgroup})")

    ax.set_xscale("symlog" if x_column == "k" else "log")
    ax.set_xticks(pivot.columns.tolist())
    ax.set_xticklabels([str(v) for v in pivot.columns.tolist()])
    ax.set_xlabel(x_label)
    ax.set_ylabel("Accuracy (Regresión Logística)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_ldiv_degradation(df_results: pd.DataFrame, baseline_k5: pd.DataFrame, output_path: Path) -> None:
    """Curva de degradación de Accuracy/F1 a lo largo del barrido de l (k=5 fijo)."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 11), sharex=True)
    ldf = df_results[df_results["tipo"] == "l_div"].copy()
    base = baseline_k5.set_index("Modelo")

    for model in df_results["Modelo"].unique():
        subset = ldf[ldf["Modelo"] == model].sort_values("l")
        base_acc = base.loc[model, "Accuracy"]
        base_f1 = base.loc[model, "F1-Score"]
        axes[0].plot(subset["l"], subset["Accuracy"], marker="o", linewidth=2, markersize=9, label=model)
        axes[0].axhline(base_acc, linestyle="--", alpha=0.4, linewidth=1)
        axes[1].plot(subset["l"], subset["F1-Score"], marker="o", linewidth=2, markersize=9, label=model)
        axes[1].axhline(base_f1, linestyle="--", alpha=0.4, linewidth=1)

    for ax in axes:
        ax.set_xticks(config.L_VALUES)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=10, loc="lower left")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy vs $l$ (con $k=5$ fijo)")
    axes[1].set_ylabel("F1-Score (ponderado)")
    axes[1].set_xlabel("$l$ (diversidad mínima en SA)")
    axes[1].set_title("F1-Score vs $l$")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_tclos_degradation(df_results: pd.DataFrame, baseline_k5: pd.DataFrame, output_path: Path) -> None:
    """Curva de degradación de Accuracy/F1 a lo largo del barrido de t (k=5 fijo)."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 11), sharex=True)
    tdf = df_results[df_results["tipo"] == "t_clos"].copy()
    base = baseline_k5.set_index("Modelo")

    for model in df_results["Modelo"].unique():
        subset = tdf[tdf["Modelo"] == model].sort_values("t", ascending=False)
        base_acc = base.loc[model, "Accuracy"]
        base_f1 = base.loc[model, "F1-Score"]
        axes[0].plot(subset["t"], subset["Accuracy"], marker="o", linewidth=2, markersize=9, label=model)
        axes[0].axhline(base_acc, linestyle="--", alpha=0.4, linewidth=1)
        axes[1].plot(subset["t"], subset["F1-Score"], marker="o", linewidth=2, markersize=9, label=model)
        axes[1].axhline(base_f1, linestyle="--", alpha=0.4, linewidth=1)

    for ax in axes:
        ax.set_xticks(config.T_VALUES)
        ax.invert_xaxis()
        ax.grid(alpha=0.3)
        ax.legend(fontsize=10, loc="lower right")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title(r"Accuracy vs $t$ (con $k=5$ fijo, $\downarrow$ más privacidad)")
    axes[1].set_ylabel("F1-Score (ponderado)")
    axes[1].set_xlabel("$t$ (umbral t-closeness)")
    axes[1].set_title("F1-Score vs $t$")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_triples_vs_singles(df_results: pd.DataFrame, baseline_k5: pd.DataFrame, output_path: Path) -> None:
    """Compara las cuatro triples con sus contrapartidas t-closeness sola sobre LR.

    Visualiza la validación empírica de Li, Li y Venkatasubramanian (2007):
    para una elección de t suficientemente estricta, t-closeness ya implica
    una forma de l-diversidad, y la triple combinación colapsa a la single.
    """
    fig, ax = plt.subplots(figsize=(11, 7))
    lr_t = df_results[(df_results["tipo"] == "t_clos") & (df_results["Modelo"] == "Regresión Logística")].sort_values("t", ascending=False)
    lr_triple = df_results[(df_results["tipo"].str.startswith("triple_")) & (df_results["Modelo"] == "Regresión Logística")].copy()
    lr_triple = lr_triple.sort_values("t", ascending=False)

    ax.plot(lr_t["t"], lr_t["Accuracy"], marker="o", linewidth=2.5, markersize=12,
            color="#1f77b4", label="t-closeness sola")
    ax.scatter(lr_triple["t"], lr_triple["Accuracy"], s=200, marker="s",
               facecolor="none", edgecolor="#d62728", linewidth=2.5, label="Triple ($k+l+t$)")

    labels = {"triple_soft": "Soft", "triple_medium": "Medium",
              "triple_hard": "Hard", "triple_extreme": "Extreme"}
    for _, row in lr_triple.iterrows():
        annotation = f"{labels[row['tipo']]}\n($l={int(row['l'])}$)"
        ax.annotate(annotation, (row["t"], row["Accuracy"]),
                    xytext=(8, 8), textcoords="offset points", fontsize=9)

    base_acc = float(baseline_k5[baseline_k5["Modelo"] == "Regresión Logística"]["Accuracy"].iloc[0])
    ax.axhline(base_acc, color="gray", linestyle="--", alpha=0.6, label=f"Baseline $k=5$ ({base_acc:.4f})")
    ax.set_xlabel("$t$ (umbral t-closeness)")
    ax.set_ylabel("Accuracy (Regresión Logística)")
    ax.set_title("Convergencia empírica: triples colapsan a t-closeness sola excepto Hard")
    ax.invert_xaxis()
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_fairness_ldiv_tclos(df_fairness: pd.DataFrame, output_path: Path) -> None:
    """Paneles apilados de fairness para los sweeps de l-diversidad y t-closeness."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 12), sharey=True)

    ldf = df_fairness[df_fairness["tipo"].isin(["k_solo", "l_div"])].copy()
    ldf["l_eff"] = ldf.apply(lambda r: 1 if r["tipo"] == "k_solo" else int(r["l"]), axis=1)
    for race_value in ldf["race"].unique():
        subset = ldf[ldf["race"] == race_value].sort_values("l_eff")
        n_value = int(subset["n"].iloc[0])
        ax1.plot(subset["l_eff"], subset["Accuracy"], marker="o", linewidth=2, markersize=9,
                 label=f"{race_value} ($n={n_value}$)")
    ax1.set_xticks([1] + config.L_VALUES)
    ax1.set_xticklabels(["$k=5$\nsolo"] + [f"$l={l}$" for l in config.L_VALUES])
    ax1.set_ylabel("Accuracy por subgrupo (LR)")
    ax1.set_title("Equidad bajo $l$-diversidad")
    ax1.legend(fontsize=9, ncol=2, loc="lower right")
    ax1.grid(alpha=0.3)

    tdf = df_fairness[df_fairness["tipo"].isin(["k_solo", "t_clos"])].copy()
    tdf["t_eff"] = tdf.apply(lambda r: 1.0 if r["tipo"] == "k_solo" else r["t"], axis=1)
    for race_value in tdf["race"].unique():
        subset = tdf[tdf["race"] == race_value].sort_values("t_eff", ascending=False)
        n_value = int(subset["n"].iloc[0])
        ax2.plot(subset["t_eff"], subset["Accuracy"], marker="o", linewidth=2, markersize=9,
                 label=f"{race_value} ($n={n_value}$)")
    ax2.set_xticks([1.0] + config.T_VALUES)
    ax2.set_xticklabels(["$k=5$\nsolo"] + [str(t).replace(".", ",") for t in config.T_VALUES])
    ax2.invert_xaxis()
    ax2.set_xlabel(r"$t$ (umbral t-closeness, $\downarrow$ más privacidad)")
    ax2.set_ylabel("Accuracy por subgrupo (LR)")
    ax2.set_title("Equidad bajo $t$-closeness")
    ax2.legend(fontsize=9, ncol=2, loc="lower right")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_mia_bars(df_mia: pd.DataFrame, output_path: Path) -> None:
    """Gráfico de barras con Attack Accuracy y Advantage por escenario."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    scenarios = df_mia["escenario"].tolist()
    x = range(len(scenarios))

    ax.bar([i - 0.2 for i in x], df_mia["attack_acc"], 0.4, label="Attack accuracy", color="C3")
    ax.bar([i + 0.2 for i in x], df_mia["attack_advantage"], 0.4, label="Attack advantage", color="C1")
    ax.axhline(0.5, ls="--", color="black", alpha=0.5, label="Aleatorio (Acc=0,5)")
    ax.axhline(0.0, ls=":", color="black", alpha=0.5)

    ax.set_xticks(list(x))
    ax.set_xticklabels(scenarios, rotation=12)
    ax.set_ylabel("Tasa")
    ax.set_title("Auditoría adversarial: MIA Black-Box sobre cada escenario")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_ldp_degradation(df_results_aggregated: pd.DataFrame, baselines: dict, output_path: Path) -> None:
    """Curva de degradación de Accuracy/F1 vs ε user-level para la LDP.

    Mismo formato que `plot_dp_degradation` (Accuracy arriba, F1 abajo)
    para facilitar la lectura cruzada en la memoria. La zona ε ≤ 10
    (comparable con la DP global) se separa visualmente de la cola
    extendida ε ∈ {20..200}.
    """
    fig, axes = plt.subplots(2, 1, figsize=(9, 10))
    epsilons = sorted(df_results_aggregated.index.get_level_values("epsilon").unique())

    for ax, mean_col, std_col, ylabel, metric_key in [
        (axes[0], "Acc_mean", "Acc_std", "Accuracy", "acc"),
        (axes[1], "F1_mean", "F1_std", "F1-Score (ponderado)", "f1"),
    ]:
        for model in ("Regresión Logística", "Naive Bayes (Gaussian)"):
            means = [df_results_aggregated.loc[(model, e), mean_col] for e in epsilons]
            stds = [df_results_aggregated.loc[(model, e), std_col] for e in epsilons]
            ax.errorbar(epsilons, means, yerr=stds, marker="o", capsize=4, label=model)

        for color, model_key in zip(("C0", "C1"), ("Regresión Logística", "Naive Bayes (Gaussian)")):
            ax.axhline(
                baselines[model_key][0 if metric_key == "acc" else 1],
                ls="--", color=color, alpha=0.5, lw=1,
                label=f"Baseline {model_key.split()[0]}",
            )

        ax.axvspan(min(epsilons), 10, alpha=0.07, color="C2",
                   label="Rango comparable con DP global")
        ax.set_xscale("log")
        ax.set_xticks(epsilons)
        ax.set_xticklabels([f"{e:g}" for e in epsilons])
        ax.set_xlabel(r"Presupuesto de privacidad por registro $\epsilon$")
        ax.set_ylabel(ylabel)
        ax.set_title(
            rf"DP Local: {ylabel} vs $\epsilon$ (mean$\pm$std, $n={config.N_REPETITIONS_DP}$)"
        )
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_ldp_vs_dp(
    df_ldp_aggregated: pd.DataFrame,
    df_dp_aggregated: pd.DataFrame,
    baseline_lr_acc: float,
    output_path: Path,
    majority_class_acc: float = None,
) -> None:
    """Comparativa directa LDP vs DP global sobre Regresión Logística.

    Ambas curvas comparten eje ε (user-level). La distancia vertical entre
    ellas a igual ε cuantifica el coste de no disponer de un curador de
    confianza; es la figura central de la sección de resultados LDP.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for df_agg, label, color, marker in (
        (df_dp_aggregated, "DP global (diffprivlib)", "C0", "o"),
        (df_ldp_aggregated, "DP local (k-RR + Laplace)", "C3", "s"),
    ):
        subset = df_agg.loc["Regresión Logística"]
        epsilons = sorted(subset.index.unique())
        means = [subset.loc[e, "Acc_mean"] for e in epsilons]
        stds = [subset.loc[e, "Acc_std"] for e in epsilons]
        ax.errorbar(epsilons, means, yerr=stds, marker=marker, capsize=4,
                    color=color, label=label)

    ax.axhline(baseline_lr_acc, ls="--", color="black", alpha=0.6, lw=1,
               label="Baseline sin privacidad")
    if majority_class_acc is not None:
        ax.axhline(majority_class_acc, ls=":", color="gray", alpha=0.6, lw=1,
                   label=f"Clasificador mayoritario ({majority_class_acc:.3f})")

    ax.set_xscale("log")
    all_epsilons = sorted(
        set(df_ldp_aggregated.index.get_level_values("epsilon"))
        | set(df_dp_aggregated.index.get_level_values("epsilon"))
    )
    ax.set_xticks(all_epsilons)
    ax.set_xticklabels([f"{e:g}" for e in all_epsilons])
    ax.set_xlabel(r"Presupuesto de privacidad por registro $\epsilon$")
    ax.set_ylabel("Accuracy (Regresión Logística)")
    ax.set_title(
        rf"DP local vs DP global a igual presupuesto (mean$\pm$std, $n={config.N_REPETITIONS_DP}$)"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
