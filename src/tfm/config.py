"""Constantes y parámetros del pipeline experimental.

Fuente única de verdad para rutas, semillas, barridos de privacidad y las
configuraciones l-diversidad / t-closeness. Ningún otro módulo debe
hardcodear rutas ni nombres de archivo de resultados.

Nota: el paquete debe instalarse en modo editable (`pip install -e .`)
para que ROOT apunte al raíz del repositorio y no a site-packages.
"""

from pathlib import Path

# Rutas relativas al raíz del repositorio (src/tfm/config.py → tres niveles)
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "arx_kit"

# Subestructura organizada de arx_kit/ (ver arx_kit/README.md)
INPUTS_DIR = DATA_DIR / "inputs"                    # arx_train, arx_test, hierarchies
ARX_OUTPUTS_DIR = DATA_DIR / "arx_outputs"          # CSVs anonimizados producidos por ARX
HIERARCHIES_DIR = INPUTS_DIR / "arx_hierarchies"
RESULTS_KANON_DIR = DATA_DIR / "results" / "kanon"
RESULTS_DP_DIR = DATA_DIR / "results" / "dp"
RESULTS_LDIV_TCLOS_DIR = DATA_DIR / "results" / "ldiv_tclos"
RESULTS_MIA_DIR = DATA_DIR / "results" / "mia"
RESULTS_LDP_DIR = DATA_DIR / "results" / "ldp"

# Figuras (PNG) consumidas por la memoria
RESULTS_DIR = ROOT / "results"

# Caché local del dataset UCI (gitignored; evita re-descargar en cada fase)
CACHE_DIR = ROOT / "data_cache"
UCI_CACHE_FILE = CACHE_DIR / "diabetes_uci_296.pkl"


def ensure_dirs() -> None:
    """Crea la estructura de directorios de resultados si no existe."""
    for directory in (
        RESULTS_DIR,
        RESULTS_KANON_DIR,
        RESULTS_DP_DIR,
        RESULTS_LDIV_TCLOS_DIR,
        RESULTS_MIA_DIR,
        RESULTS_LDP_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


# Reproducibilidad
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Barridos de privacidad
K_VALUES = [2, 5, 10, 25, 50]
EPSILON_VALUES = [0.1, 0.5, 1.0, 5.0, 10.0]
N_REPETITIONS_DP = 20

# Semillas dispersas para las réplicas de Privacidad Diferencial.
# Se eligen valores no consecutivos para minimizar correlaciones residuales del
# generador pseudoaleatorio (Mersenne Twister). El número total coincide con
# N_REPETITIONS_DP.
DP_SEEDS = [42, 137, 271, 314, 1729, 2718, 3141, 6022, 8128, 9999,
            10007, 11113, 13121, 17171, 19273, 23131, 29327, 31193, 33391, 37397]

# Extensión a l-diversidad y t-closeness (k fijado en 5 por convención clínica)
L_VALUES = [2, 3, 5]
T_VALUES = [0.5, 0.4, 0.35, 0.3, 0.25]
SENSITIVE_ATTRIBUTE = "diag_1_category"

# Privacidad Diferencial Local (Fase 12).
#
# El presupuesto es POR REGISTRO (user-level): se reparte uniformemente entre
# los atributos perturbables mediante composición secuencial (ε_j = ε / n_eff).
# Los cinco primeros valores replican EPSILON_VALUES para la comparativa
# directa con la DP global; la cola extendida {20, 50, 100, 200} explora a
# partir de qué presupuesto —ya sin valor protector real— la utilidad
# comienza a recuperarse, localizando la frontera empírica de viabilidad
# de la LDP sobre este dataset (n ≈ 78k, 45 atributos).
#
# Lectura alternativa "por atributo" (práctica industrial tipo RAPPOR/Apple):
# equivale a reparametrizar el eje, ε_atributo = ε_registro / n_eff; no
# requiere experimento aparte y se discute como tal en la memoria.
LDP_EPSILON_VALUES = [0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]

# Columnas con código entero pero semántica nominal (catálogos administrativos
# del data dictionary IDS_mapping): se perturban con k-RR sobre su dominio
# observado, aunque para el modelo sigan entrando como variables numéricas
# (mismo tratamiento de columna que en baseline/DP global → column space 121).
LDP_NOMINAL_INT_COLUMNS = (
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
)


def arx_output_filename(k: int, l: int = None, t: float = None) -> str:
    """Nombre canónico del CSV exportado desde ARX Desktop.

    Convención: arx_output_k<k>[_l<l>][_t<t sin punto>].csv
    Ejemplos: k=10 → arx_output_k10.csv; k=5, t=0.25 → arx_output_k5_t025.csv
    """
    name = f"arx_output_k{k}"
    if l is not None:
        name += f"_l{l}"
    if t is not None:
        name += f"_t{str(t).replace('.', '')}"
    return name + ".csv"


# Las 13 configuraciones de la Fase 8: (archivo, k, l, t, tipo).
# Mapeo canónico entre archivo y parámetros; las fases 9 y 11 seleccionan
# subconjuntos de esta lista.
LT_CONFIGURATIONS = [
    ("arx_output_k5.csv",          5, None, None, "k_solo"),
    ("arx_output_k5_l2.csv",       5, 2,    None, "l_div"),
    ("arx_output_k5_l3.csv",       5, 3,    None, "l_div"),
    ("arx_output_k5_l5.csv",       5, 5,    None, "l_div"),
    ("arx_output_k5_t05.csv",      5, None, 0.5,  "t_clos"),
    ("arx_output_k5_t04.csv",      5, None, 0.4,  "t_clos"),
    ("arx_output_k5_t035.csv",     5, None, 0.35, "t_clos"),
    ("arx_output_k5_t03.csv",      5, None, 0.3,  "t_clos"),
    ("arx_output_k5_t025.csv",     5, None, 0.25, "t_clos"),
    ("arx_output_k5_l2_t05.csv",   5, 2,    0.5,  "triple_soft"),
    ("arx_output_k5_l3_t035.csv",  5, 3,    0.35, "triple_medium"),
    ("arx_output_k5_l5_t03.csv",   5, 5,    0.3,  "triple_hard"),
    ("arx_output_k5_l5_t025.csv",  5, 5,    0.25, "triple_extreme"),
]

# Fase 9 — escenarios MIA sobre las configuraciones l/t más estrictas.
# El orden importa: la semilla de cada escenario se toma de DP_SEEDS por índice.
MIA_LT_TARGETS = [
    ("k=5 (referencia)",          "arx_output_k5.csv"),
    ("k=5 + l=5",                 "arx_output_k5_l5.csv"),
    ("k=5 + t=0.25",              "arx_output_k5_t025.csv"),
    ("k=5 + l=5 + t=0.3 (Hard)",  "arx_output_k5_l5_t03.csv"),
]

# Fase 11 — selección razonada de configuraciones para McNemar (subconjunto
# de LT_CONFIGURATIONS; el resto se omite por redundancia con singles ya
# testeados o por efecto por debajo del umbral de detección útil).
MCNEMAR_LT_CONFIGURATIONS = [
    ("k=5 + l=5",                "arx_output_k5_l5.csv"),
    ("k=5 + t=0.5",              "arx_output_k5_t05.csv"),
    ("k=5 + t=0.25",             "arx_output_k5_t025.csv"),
    ("k=5 + l=5 + t=0.3 (Hard)", "arx_output_k5_l5_t03.csv"),
]

# Identificadores y atributos del dataset clínico
DATASET_ID = 296  # Diabetes 130-US hospitals (UCI ML Repo)

QID_COLUMNS = [
    "race",
    "gender",
    "age",
    "time_in_hospital",
    "admission_type_id",
]

# QIDs numéricos que ARX trata como categóricos: deben convertirse a string
# antes de cualquier operación que compare train anonimizado con test en claro.
QID_NUMERIC_AS_STR = ("admission_type_id", "time_in_hospital")

TARGET_COLUMN = "readmitted"

DROP_COLUMNS_FOR_ANONYMIZATION = ["diag_1", "diag_2", "diag_3", "medical_specialty"]

IMPUTE_COLUMNS = ["max_glu_serum", "A1Cresult", "medical_specialty", "payer_code"]
IMPUTE_VALUE = "No_Registrado"

DROP_NA_SUBSET = ["race", "diag_1", "diag_2", "diag_3"]
