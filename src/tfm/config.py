"""Constantes y parámetros del pipeline experimental: única fuente de verdad
para rutas, semillas y barridos. Instalar en editable (`pip install -e .`)
para que ROOT apunte al raíz del repositorio."""

from pathlib import Path

# Rutas relativas al raíz del repositorio (tres niveles desde este archivo)
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

# Semillas dispersas (no consecutivas, minimizan correlaciones del PRNG)
# compartidas por DP, LDP y réplicas MIA; tantas como N_REPETITIONS_DP.
DP_SEEDS = [42, 137, 271, 314, 1729, 2718, 3141, 6022, 8128, 9999,
            10007, 11113, 13121, 17171, 19273, 23131, 29327, 31193, 33391, 37397]

# Extensión a l-diversidad y t-closeness (k fijado en 5 por convención clínica)
L_VALUES = [2, 3, 5]
T_VALUES = [0.5, 0.4, 0.35, 0.3, 0.25]
SENSITIVE_ATTRIBUTE = "diag_1_category"

# LDP (Fase 12): presupuesto POR REGISTRO repartido uniformemente entre
# atributos. Los cinco primeros valores replican EPSILON_VALUES; la cola
# extendida localiza la frontera empírica de recuperación de utilidad.
LDP_EPSILON_VALUES = [0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]

# Columnas enteras con semántica nominal (catálogos administrativos):
# se perturban con k-RR sobre su dominio observado.
LDP_NOMINAL_INT_COLUMNS = (
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
)


def arx_output_filename(k: int, l: int = None, t: float = None) -> str:
    """Nombre canónico del CSV de ARX: arx_output_k<k>[_l<l>][_t<t sin punto>].csv"""
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

# Fase 13: auditoría MIA reforzada (réplicas del Black-Box + LiRA offline).

# Tier 1: réplicas del ataque de la fase 6 con las semillas de DP_SEEDS.
MIA_N_REPLICAS = 20

# Víctimas del Tier 1; en los escenarios DP solo aplica la LR
# (diffprivlib no ofrece un Random Forest privado).
MIA_VICTIMS = ("Regresión Logística", "Random Forest")

# Escenarios l/t estrictos del Tier 1 (sin la referencia k=5, ya cubierta).
MIA_REPLICAS_LT_TARGETS = [
    ("k=5 + l=5",                 "arx_output_k5_l5.csv"),
    ("k=5 + t=0.25",              "arx_output_k5_t025.csv"),
    ("k=5 + l=5 + t=0.3 (Hard)",  "arx_output_k5_l5_t03.csv"),
]

# Tier 2: ataque LiRA offline (Carlini et al., 2022) con modelos sombra.
MIA_SHADOW_COUNT = 64           # modelos sombra por escenario
LIRA_SHADOW_FRACTION = 0.5      # fracción de la población por sombra
LIRA_N_TARGETS_PER_CLASS = 10000  # objetivos member / non-member muestreados
LIRA_FPR_TARGETS = [0.001, 0.01, 0.1]  # TPR @ FPR bajo (métrica clave)
LIRA_MIN_OUT_SHADOWS = 8        # mínimo de sombras OUT; por debajo, sigma global

# Escenarios del Tier 2 sobre la víctima LR (LiRA sobre RF y la variante
# online quedan como ampliación opcional).
LIRA_SCENARIOS = [
    ("Baseline LR",  "baseline", None),
    ("LR · k=10",    "kanon",    10),
    ("LR · k=50",    "kanon",    50),
    ("LR · DP ε=1.0", "dp",      1.0),
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
