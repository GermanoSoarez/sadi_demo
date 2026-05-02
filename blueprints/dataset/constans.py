# blueprints/dataset/constans.py

# =========================================================
# TIPOS DE DATASET
# =========================================================

DATASET_TYPE_CHOICES = [
    ("dataset", "Dataset general"),
    ("multivariate", "Multivariate"),
    ("survey_normal", "Encuesta normal"),
    ("survey_likert_5", "Likert 5 puntos"),
    ("survey_likert_7", "Likert 7 puntos"),
]

DATASET_TYPE_LABELS = dict(DATASET_TYPE_CHOICES)
DATASET_TYPE_VALUES = {value for value, _ in DATASET_TYPE_CHOICES}

DATASET_TYPE_BADGES = {
    "dataset": "badge-info",
    "multivariate": "badge-info",
    "survey_normal": "badge-ok",
    "survey_likert_5": "badge-ok",
    "survey_likert_7": "badge-ok",
}

# =========================================================
# AREAS DE INVESTIGACION
# =========================================================

RESEARCH_AREA_CHOICES = [
    ("general", "General"),
    ("biomedicina", "Salud / Biomedicina"),
    ("educacion", "Educación"),
    ("finanzas", "Economía y Finanzas"),
    ("marketing", "Marketing / Mercado"),
    ("agronomia", "Agronomía"),
    ("social", "Ciencias Sociales"),
    ("ingenieria", "Ingeniería / Tecnología"),
    ("medio_ambiente", "Medio Ambiente"),
    ("legal", "Jurídico"),
    ("psicologia", "Psicología"),
]

RESEARCH_AREA_LABELS = dict(RESEARCH_AREA_CHOICES)
RESEARCH_AREA_VALUES = {value for value, _ in RESEARCH_AREA_CHOICES}

RESEARCH_AREA_BADGES = {
    "general": "badge-warn",
    "biomedicina": "badge-ok",
    "educacion": "badge-info",
    "finanzas": "badge-info",
    "marketing": "badge-info",
    "agronomia": "badge-info",
    "social": "badge-info",
    "ingenieria": "badge-info",
    "medio_ambiente": "badge-info",
    "legal": "badge-info",
    "psicologia": "badge-ok",
}

# =========================================================
# HELPERS
# =========================================================

def normalize_dataset_type(value: str | None) -> str:
    value = (value or "").strip()
    return value if value in DATASET_TYPE_VALUES else "dataset"

def normalize_research_area(value: str | None) -> str:
    value = (value or "").strip()
    return value if value in RESEARCH_AREA_VALUES else "general"

def dataset_type_label(value: str | None) -> str:
    value = normalize_dataset_type(value)
    return DATASET_TYPE_LABELS.get(value, "Dataset general")

def research_area_label(value: str | None) -> str:
    value = normalize_research_area(value)
    return RESEARCH_AREA_LABELS.get(value, "General")

def dataset_type_badge(value: str | None) -> str:
    value = normalize_dataset_type(value)
    return DATASET_TYPE_BADGES.get(value, "badge-info")

def research_area_badge(value: str | None) -> str:
    value = normalize_research_area(value)
    return RESEARCH_AREA_BADGES.get(value, "badge-info")

def dataset_detail_endpoint(value: str | None) -> str:
    value = normalize_dataset_type(value)
    if value == "multivariate":
        return "multivariate.multivariate_detail"
    return "dataset.dataset_detail"