# Guía paso a paso · k-Anonimidad con ARX Desktop

Esta guía resume el procedimiento manual para producir las cinco versiones
anonimizadas (`arx_output_k2.csv`, `arx_output_k5.csv`, …, `arx_output_k50.csv`)
que la fase 3 del pipeline (`tfm run 03`) consume después.

---

## 0 · Preparación

1. Comprueba que tienes los archivos generados por la fase 2 (`tfm run 02`)
   en `arx_kit/inputs/`:
   - `arx_train.csv` (78.442 filas, separador `;`)
   - `arx_test.csv` (19.611 filas, separador `;`) — no se anonimiza
   - `arx_hierarchies/race.csv`
   - `arx_hierarchies/gender.csv`
   - `arx_hierarchies/age.csv`
   - `arx_hierarchies/time_in_hospital.csv`
   - `arx_hierarchies/admission_type_id.csv`

2. Descarga ARX Desktop:
   - Web oficial: <https://arx.deidentifier.org/downloads/>
   - Requiere Java 11+ (instala Temurin JDK si no lo tienes).
   - Usa la versión **3.9.1** o superior.

3. Lanza ARX (`arx-3.9.1.jar` o el ejecutable de tu plataforma).

---

## 1 · Importar el conjunto de datos

1. Menú: **File → Import data → CSV**.
2. Selecciona `arx_train.csv`.
3. En el diálogo de importación:
   - **Separator** → `Semicolon (;)`
   - **Charset** → `UTF-8`
   - **First row contains column names** → ✓
4. Pulsa **Next** → comprueba que detecta **46 columnas** y **78.442 filas** → **Finish**.

---

## 2 · Configurar atributos

En la pestaña **Configure → Attributes** asigna a cada columna su tipo:

| Columna | Tipo | Acción |
|---|---|---|
| `race` | **Quasi-identifying** | Cargar jerarquía `arx_hierarchies/race.csv` |
| `gender` | **Quasi-identifying** | Cargar jerarquía `arx_hierarchies/gender.csv` |
| `age` | **Quasi-identifying** | Cargar jerarquía `arx_hierarchies/age.csv` |
| `time_in_hospital` | **Quasi-identifying** | Cargar jerarquía `arx_hierarchies/time_in_hospital.csv` |
| `admission_type_id` | **Quasi-identifying** | Cargar jerarquía `arx_hierarchies/admission_type_id.csv` |
| `readmitted` | **Insensitive** | Variable objetivo — se protege en la fase de DP, no aquí |
| Resto (40 columnas) | **Insensitive** | — |

> ⚠️ Si marcas `readmitted` como **Sensitive**, ARX exigirá un criterio adicional (l-diversidad o t-cercanía) y lanzará el error *"No privacy model specified for sensitive attribute: 'readmitted'"*. En este experimento solo aplicamos k-anonimidad sobre los QIDs, así que `readmitted` debe quedar como **Insensitive**.

> Para cargar una jerarquía: clic derecho sobre la columna → **Hierarchy → Load
> from file** → CSV separado por `;`. ARX detectará automáticamente los niveles.

> ⚠️ Si ARX se queja de un valor sin asignar (típicamente
> `Unknown/Invalid` en `gender`), revisa que estás usando los archivos del
> kit corregido — los originales del notebook v2 omitían ese valor.

---

## 3 · Configurar criterio de privacidad

En la pestaña **Configure → Privacy criteria** añade:

1. **Add → k-Anonymity** y fija el valor de **k**.
   - Para la primera pasada usa `k = 2`.
2. Deja el resto de criterios sin marcar (l-diversidad y t-cercanía se
   tratan en una iteración futura si la curva de utilidad lo justifica).

---

## 4 · Configurar utilidad y supresión

Pestaña **Configure → Utility / Suppression**:

- **Utility measure** → `Loss` (default).
- **Suppression limit** → `100%` (permite que ARX suprima localmente lo que
  haga falta para no generalizar excesivamente).
- **Generalization aggregation** → `Sum` (default).

---

## 5 · Anonymize!

1. Pulsa el botón **Anonymize!** (icono de candado).
2. ARX construye el retículo de generalizaciones y devuelve la solución
   óptima (suele tardar entre 5 y 60 segundos según `k`).
3. Cuando termine, ve a la pestaña **Explore results**:
   - Inspecciona el lattice y el nivel de generalización aplicado a cada QID.
   - En **Analyze utility / Risk**: anota el porcentaje de filas suprimidas
     y la métrica Loss para tu memoria.

---

## 6 · Exportar

1. Pestaña **Analyze → Output data** → **Apply**.
2. Menú: **File → Export data**.
3. Formato CSV con separador `;`.
4. Nombre: `arx_output_k<valor>.csv` (p. ej. `arx_output_k2.csv`).

---

## 7 · Repetir para los cinco valores de k

Repite los pasos 3 → 6 cambiando solo el valor de **k**:

| Iteración | k | Archivo de salida |
|-----------|---|-------------------|
| 1 | 2 | `arx_output_k2.csv` |
| 2 | 5 | `arx_output_k5.csv` |
| 3 | 10 | `arx_output_k10.csv` |
| 4 | 25 | `arx_output_k25.csv` |
| 5 | 50 | `arx_output_k50.csv` |

> No es necesario reimportar los datos: ARX conserva la configuración entre
> iteraciones. Solo modifica el k en `Privacy criteria` y vuelve a pulsar
> **Anonymize!**.

---

## 8 · Colocar los resultados y evaluar

1. Guarda los cinco archivos `arx_output_k*.csv` en `arx_kit/arx_outputs/`.
2. Ejecuta `tfm run 03`: el loop detecta automáticamente los archivos
   disponibles y produce la tabla de degradación + las figuras
   `comparativa_kanon.png` y `fairness_kanon.png`.

---

## 9 · Métricas que conviene apuntar para la memoria

Para cada k, anota desde la pestaña **Analyze utility / Risk** de ARX:

- **Suppression** (% de filas suprimidas).
- **Loss** (métrica de utilidad ARX, 0 = sin pérdida).
- **Generalization levels** seleccionados por QID.
- **Records at risk** (% de pacientes vulnerables tras anonimizar — debería
  ser 0 si k se cumple estrictamente).

Estas métricas son las que rellenan la tabla del Capítulo 4 (Resultados)
junto con las accuracies/F1 que produce el pipeline.

---

## 10 · Troubleshooting

| Síntoma | Causa probable | Solución |
|---------|----------------|----------|
| ARX no carga la jerarquía | Separador erróneo | Verifica que la jerarquía usa `;` y UTF-8 |
| Error "value not in hierarchy" | Falta un valor en el archivo `.csv` de jerarquía | Compara `df_train_raw[col].unique()` con la primera columna de la jerarquía |
| Suprime más del 90 % de filas con k bajo | Demasiados QIDs únicos en combinación | Reduce el número de QIDs o añade más niveles a las jerarquías |
| Java OutOfMemory | Heap insuficiente | Lanza ARX con `java -Xmx8g -jar arx-3.9.1.jar` |
