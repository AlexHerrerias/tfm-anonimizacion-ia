# arx_outputs/

CSVs anonimizados producidos por **ARX Desktop** a partir de `arx_kit/inputs/arx_train.csv`.

Estos archivos **sí se versionan** (~280 MB en total) de forma deliberada:
permiten ejecutar las fases 3–11 del pipeline sin pasar por ARX Desktop
(la anonimización es el único paso manual del estudio). Para regenerarlos:

1. Abrir ARX Desktop con el proyecto `inputs/tfm-arx_k2.deid` o configurando manualmente.
2. Cargar `inputs/arx_train.csv` y las jerarquías de `inputs/arx_hierarchies/`.
3. Aplicar los criterios de privacidad (ver guía `docs/guia_arx.md`).
4. Exportar cada resultado anonimizado a esta carpeta con la convención:

```
arx_output_k<K>.csv                       # k-anonimidad pura
arx_output_k<K>_l<L>.csv                  # k-anonimidad + l-diversidad
arx_output_k<K>_t<NN>.csv                 # k-anonimidad + t-closeness (t=0.NN, ej t02=0.2)
arx_output_k<K>_l<L>_t<NN>.csv            # triple combinación
```

(La misma convención está codificada en `tfm.config.arx_output_filename`.)

## Lista esperada para reproducir el TFM

**Fase k-anonimidad** (Cap. 4 §4.1):
- `arx_output_k2.csv`, `arx_output_k5.csv`, `arx_output_k10.csv`, `arx_output_k25.csv`, `arx_output_k50.csv`

**Fase l-diversidad / t-closeness** (Cap. 4 §4.4), con `diag_1_category` marcada como Sensitive:
- l-div: `arx_output_k5_l2.csv`, `arx_output_k5_l3.csv`, `arx_output_k5_l5.csv`
- t-clos: `arx_output_k5_t05.csv`, `arx_output_k5_t04.csv`, `arx_output_k5_t035.csv`, `arx_output_k5_t03.csv`, `arx_output_k5_t025.csv`, `arx_output_k5_t02.csv` (evidencia de frontera empírica)
- Triples: `arx_output_k5_l2_t05.csv`, `arx_output_k5_l3_t035.csv`, `arx_output_k5_l5_t03.csv`, `arx_output_k5_l5_t025.csv`

(La correspondencia canónica entre archivos y parámetros está definida en `tfm.config.LT_CONFIGURATIONS`.)
