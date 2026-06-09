# arx_kit — Artefactos experimentales del TFM

Carpeta organizada en tres bloques: entradas a ARX, salidas anonimizadas y resultados de los experimentos Python.

```
arx_kit/
├── inputs/                          # Entradas a ARX Desktop
│   ├── arx_train.csv                # 78.442 registros para anonimizar
│   ├── arx_test.csv                 # 19.611 registros en claro (test set)
│   ├── arx_hierarchies/             # Jerarquías de generalización por QID
│   ├── tfm-arx.deid                 # Proyecto ARX de ejemplo
│   └── tfm-arx_k2.deid              # Proyecto ARX guardado (k=2 ejemplo)
│       (procedimiento manual de la GUI: docs/guia_arx.md)
│
├── arx_outputs/                     # CSVs anonimizados exportados desde ARX
│   ├── arx_output_k{2,5,10,25,50}.csv             # k-anonimidad pura
│   ├── arx_output_k5_l{2,3,5}.csv                 # l-diversidad sobre k=5
│   ├── arx_output_k5_t{02,025,03,035,04,05}.csv   # t-closeness sobre k=5
│   └── arx_output_k5_l*_t*.csv                    # Combinaciones triples
│
└── results/                         # Outputs del pipeline Python (tfm run …)
    ├── kanon/                       # Fase ARX (k-anonimidad)
    │   ├── perfil_privacidad.csv    #   Fase 0 — riesgo basal de reidentificación
    │   ├── baseline.csv             #   Fase 1
    │   ├── resultados_kanon.csv     #   Fase 3
    │   ├── fairness_kanon.csv       #   Fase 3
    │   ├── loss_disparity_kanon.csv #   Fase 3
    │   └── mcnemar_kanon.csv        #   Fase 7
    ├── dp/                          # Fase Privacidad Diferencial (n=20)
    │   ├── resultados_dp.csv        #   Fase 4
    │   ├── fairness_dp.csv          #   Fase 4
    │   ├── resultados_combo.csv     #   Fase 5
    │   ├── fairness_combo.csv       #   (generado por el notebook v3; sin script que lo regenere)
    │   └── wilcoxon_dp.csv          #   Fase 7
    ├── ldiv_tclos/                  # Extensión l-diversidad / t-closeness
    │   ├── resultados_ldiv_tclos.csv    # Fase 8
    │   ├── fairness_ldiv_tclos.csv      # Fase 8
    │   ├── verificacion_ldiv_tclos.csv  # Fase 8
    │   └── mcnemar_ldiv_tclos.csv       # Fase 11
    └── mia/                         # Auditoría MIA Black-Box
        ├── mia_results.csv          #   Fase 6
        └── mia_ldiv_tclos.csv       #   Fase 9
```

## Convención de nombres

- `arx_output_kK.csv` → k-anonimidad con valor K
- `arx_output_kK_lL.csv` → k-anonimidad + l-diversidad (l=L)
- `arx_output_kK_tNN.csv` → k-anonimidad + t-closeness (t=0.NN, p.ej. t02 = 0.2)
- `arx_output_kK_lL_tNN.csv` → triple combinación
