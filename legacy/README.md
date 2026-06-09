# legacy — Material histórico no mantenido

## TFM_pipeline.ipynb

Notebook orquestador original para Google Colab, congelado en junio de 2026
al consolidarse el pipeline en el paquete Python `tfm` (`tfm run all`). La
primera celda del notebook detalla los motivos del archivado (duplicación de
lógica, cobertura parcial de las fases y rutas relativas que fallaban fuera
de Colab).

Información que solo documentaba este notebook:

- Repositorio público en GitHub usado por el bootstrap de Colab:
  <https://github.com/alexherrland/tfm-anonimizacion-ia>
- El perfilado de privacidad (riesgo basal de reidentificación) era la única
  lógica exclusiva del notebook; ahora es la fase 00 del pipeline
  (`tfm/profiling.py`).

## Artefactos de versiones anteriores

Los siguientes archivos de resultados proceden de una versión anterior del
notebook (v3) y ningún script actual los regenera; se conservan porque la
memoria puede referenciarlos:

- `arx_kit/results/dp/fairness_combo.csv` (versionado en git)
- `results/comparativa_combo.png`, `results/fairness_combo.png` y
  `results/comparativa_dp_vs_kanon.png` (solo copias locales, fuera de git)
