"""CLI del pipeline experimental: `tfm run <fase...>` / `tfm run all` / `tfm list`."""

import argparse
import importlib
import sys
import time

# Registro ordenado de fases: id → (módulo, descripción corta)
PHASES = {
    "00": ("tfm.pipeline.phase00_profiling", "Perfilado de privacidad (riesgo basal de reidentificación)"),
    "01": ("tfm.pipeline.phase01_baseline", "Baseline de utilidad sin privacidad"),
    "02": ("tfm.pipeline.phase02_export_arx", "Exportación de CSVs y jerarquías para ARX Desktop"),
    "03": ("tfm.pipeline.phase03_evaluate_kanon", "Evaluación post-ARX (k-anonimidad) + fairness"),
    "04": ("tfm.pipeline.phase04_dp_sweep", "Barrido de Privacidad Diferencial (ε × 20 reps)"),
    "05": ("tfm.pipeline.phase05_combo", "Defensa en profundidad: DP sobre k=10"),
    "06": ("tfm.pipeline.phase06_mia", "Auditoría adversarial MIA Black-Box"),
    "07": ("tfm.pipeline.phase07_statistical_tests", "Tests de significancia (McNemar + Wilcoxon)"),
    "08": ("tfm.pipeline.phase08_eval_ldiv_tclos", "Evaluación l-diversidad / t-closeness / triples"),
    "09": ("tfm.pipeline.phase09_mia_ldiv_tclos", "MIA sobre configuraciones l/t estrictas"),
    "10": ("tfm.pipeline.phase10_plots_ldiv_tclos", "Figuras de la extensión l/t"),
    "11": ("tfm.pipeline.phase11_mcnemar_ldiv_tclos", "McNemar sobre configuraciones l/t extremas"),
}

# La Fase 2 termina en un paso manual (ARX Desktop). Tras ejecutarla, las
# fases 3+ requieren que los CSVs exportados existan en arx_kit/arx_outputs/.
MANUAL_STEP_AFTER = "02"


def _normalize(phase: str) -> str:
    """Acepta '3', '03' o 'phase03' y devuelve el id canónico de dos dígitos."""
    digits = "".join(ch for ch in phase if ch.isdigit())
    if not digits:
        return phase
    return f"{int(digits):02d}"


def _run_phase(phase_id: str) -> None:
    module_name, description = PHASES[phase_id]
    print(f"\n{'=' * 70}")
    print(f"Fase {phase_id} — {description}")
    print("=" * 70)
    start = time.time()
    module = importlib.import_module(module_name)
    module.run()
    print(f"[Fase {phase_id} completada en {time.time() - start:.1f} s]")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="tfm",
        description="Pipeline del TFM de anonimización y privacidad en IA sanitaria.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Ejecuta una o varias fases, o `all`")
    run_parser.add_argument(
        "phases",
        nargs="+",
        help="Ids de fase (ej. 01, 04 09) o `all` para la secuencia completa",
    )

    subparsers.add_parser("list", help="Lista las fases disponibles")

    args = parser.parse_args(argv)

    if args.command == "list":
        for phase_id, (_, description) in PHASES.items():
            print(f"  {phase_id}  {description}")
        return 0

    if args.phases == ["all"]:
        selected = list(PHASES)
    else:
        selected = [_normalize(p) for p in args.phases]
        unknown = [p for p in selected if p not in PHASES]
        if unknown:
            print(f"Fases desconocidas: {', '.join(unknown)}. Usa `tfm list`.")
            return 2

    for phase_id in selected:
        _run_phase(phase_id)
        if phase_id == MANUAL_STEP_AFTER and len(selected) > 1:
            print(
                "\nNota: la anonimización en ARX Desktop es manual (docs/guia_arx.md). "
                "Las fases siguientes usan los CSVs ya presentes en arx_kit/arx_outputs/."
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
