# TFM — Anonimización y Privacidad en IA Sanitaria

Repositorio con el código que acompaña al Trabajo Fin de Máster
*Estrategias de anonimización y privacidad en el ciclo de vida del
aprendizaje automático: un marco metodológico aplicado a la predicción
de readmisión hospitalaria*.

El estudio cuantifica el equilibrio entre privacidad y utilidad predictiva
en un caso de uso clínico, contrastando $k$-anonimidad (ARX), Privacidad
Diferencial (`diffprivlib`) y su combinación, junto con una auditoría
adversarial mediante MIA (`adversarial-robustness-toolbox`).

## Estructura del repositorio

```
.
├── pyproject.toml             Paquete instalable `tfm` + comando de consola
├── src/tfm/                   Módulos del pipeline
│   ├── config.py              Constantes, rutas y configuraciones (fuente única)
│   ├── data_loader.py         Ingesta UCI (con caché local), limpieza, CIE-9
│   ├── preprocessing.py       Split, OHE, escalado, percentiles para DP
│   ├── arx_io.py              Carga unificada de CSVs exportados por ARX
│   ├── profiling.py           Riesgo basal de reidentificación (fase 00)
│   ├── models.py              Constructores de modelos baseline y DP
│   ├── kanon.py               Soporte ARX (exportar CSV, jerarquías, evaluar)
│   ├── ldiv_tclos.py          l-diversidad, t-closeness y triples (k+l+t)
│   ├── differential_privacy.py  Sweep ε y combinación k+DP
│   ├── local_dp.py            Privacidad Diferencial Local (k-RR + Laplace)
│   ├── fairness.py            Disparidad por subgrupo race
│   ├── statistical_tests.py   McNemar y Wilcoxon (con Bonferroni)
│   ├── mia.py                 Membership Inference Attack Black-Box
│   ├── plotting.py            Figuras de la memoria
│   └── pipeline/              Fases 00–11 + CLI (`tfm run …`)
├── arx_kit/                   Entradas, salidas y resultados experimentales
│   ├── inputs/                CSVs y jerarquías que consume ARX Desktop
│   ├── arx_outputs/           CSVs anonimizados exportados desde ARX (versionados)
│   └── results/               CSVs de resultados del pipeline (versionados)
├── results/                   Figuras (PNG) de la memoria (no versionadas)
├── docs/guia_arx.md           Guía paso a paso de ARX Desktop (único paso manual)
├── legacy/                    Notebook Colab original, congelado (ver legacy/README.md)
├── requirements.txt           Espejo de las dependencias del paquete
└── TFM_Memory.pdf             Memoria del TFM
```

## Instalación

Requisitos: Python 3.10–3.12 y, solo para la fase de anonimización manual,
ARX Desktop (<https://arx.deidentifier.org/>).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

La instalación editable registra el comando `tfm` y ancla las rutas del
pipeline al raíz del repositorio. Ya no es necesario exportar `PYTHONPATH`.

## Ejecución del pipeline

El pipeline se compone de **trece fases** (00–12). Lista de fases:

```bash
tfm list
```

Ejecución completa o por fases:

```bash
tfm run all          # las trece fases en secuencia
tfm run 04           # una fase concreta
tfm run 08 09 10 11  # varias fases en orden
```

| Fase | Qué hace | Salidas principales |
|------|----------|---------------------|
| 00 | Perfilado de privacidad (riesgo basal de reidentificación) | `perfil_privacidad.csv` |
| 01 | Baseline de utilidad sin privacidad | `baseline.csv` |
| 02 | Exporta CSVs y jerarquías para ARX Desktop | `arx_kit/inputs/*` |
| —  | **Paso manual**: anonimizar en ARX Desktop ([docs/guia_arx.md](docs/guia_arx.md)) | `arx_kit/arx_outputs/*.csv` |
| 03 | Utilidad post-ARX + fairness por `race` | `resultados_kanon.csv`, `fairness_kanon.csv`, `loss_disparity_kanon.csv` |
| 04 | Barrido de ε (20 repeticiones) | `resultados_dp.csv`, `fairness_dp.csv` |
| 05 | Defensa en profundidad k=10 + DP | `resultados_combo.csv` |
| 06 | Auditoría adversarial MIA (6 escenarios) | `mia_results.csv` |
| 07 | McNemar (k-anon) + Wilcoxon (DP), Bonferroni | `mcnemar_kanon.csv`, `wilcoxon_dp.csv` |
| 08 | Utilidad/equidad/verificación l-div, t-clos y triples | `resultados_ldiv_tclos.csv`, `fairness_ldiv_tclos.csv`, `verificacion_ldiv_tclos.csv` |
| 09 | MIA sobre configuraciones l/t estrictas | `mia_ldiv_tclos.csv` |
| 10 | Figuras de la extensión l/t | 4 PNG en `results/` |
| 11 | McNemar sobre 4 configuraciones l/t extremas | `mcnemar_ldiv_tclos.csv` |
| 12 | Barrido de Privacidad Diferencial **Local** (LDP, ε × 20 reps) | `resultados_ldp.csv`, `fairness_ldp.csv`, `wilcoxon_ldp.csv` |

Los CSVs de resultados se guardan bajo `arx_kit/results/` y las figuras PNG
en `results/`; los nombres coinciden con los referenciados desde la memoria.

El dataset UCI se descarga por red la primera vez y queda cacheado en
`data_cache/` (gitignored); borra esa carpeta para forzar una descarga limpia.

## Reproducibilidad

Todos los componentes estocásticos están parametrizados por la semilla
`RANDOM_STATE = 42` definida en `tfm/config.py`. El barrido de Privacidad
Diferencial utiliza **veinte repeticiones independientes** con semillas
dispersas (`42, 137, 271, 314, 1729, 2718, 3141, 6022, 8128, 9999, 10007,
11113, 13121, 17171, 19273, 23131, 29327, 31193, 33391, 37397`), elegidas
para minimizar correlaciones residuales del generador Mersenne Twister.
Con `n=20` el test de Wilcoxon de una muestra alcanza significancia
estadística formal frente al baseline en todas las celdas con
degradación consistente.

Los modelos lineales y bayesianos (Regresión Logística, GaussianNB)
reproducen los valores de la memoria de forma exacta; los modelos de árbol
(Random Forest, Árbol de Decisión) pueden variar en el tercer decimal entre
entornos por la resolución de empates en los splits, sensible a la versión
de numpy/BLAS.

## Dataset

Diabetes 130-US hospitals for years 1999-2008 (UCI Machine Learning
Repository, ID 296). El conjunto se descarga automáticamente mediante
`ucimlrepo` en la primera ejecución.

## Notebook legacy

El notebook original de Google Colab (`legacy/TFM_pipeline.ipynb`) quedó
congelado al consolidarse el pipeline en el paquete `tfm`; ver
[legacy/README.md](legacy/README.md).

## Cita

Trabajo Fin de Máster, Máster Universitario en Ingeniería y Ciencia de
Datos, UNED.
