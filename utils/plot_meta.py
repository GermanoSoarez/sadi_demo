from __future__ import annotations

import os
import re


def _fname(p: str) -> str:
    return str(p).replace("\\", "/").split("/")[-1].lower()


def classify_plot_tag(p: str) -> str:
    f = _fname(p)

    if "_missing" in f:
        return "Calidad de datos"
    if "_box" in f:
        return "Descriptivo"
    if "_hist_" in f:
        return "Descriptivo"
    if "_bar_" in f:
        return "Descriptivo"
    if "_corr_" in f or "_heatmap" in f:
        return "Relacional"
    if "_scatter_" in f:
        return "Relacional"
    if "_group_mean_" in f:
        return "Comparativo"
    if "_likert_" in f or "_cronbach" in f or "_divergent" in f:
        return "Psicométrico"
    if "_factor_" in f or "_scree_" in f:
        return "Factorial"
    return "Analítico"
    if clean == "dimension_summary_scores" or clean == "dimension_radar":
            return "Comparativo"


def prettify_plot_title(p: str) -> str:
    f = _fname(p)

    if "_missing" in f:
        return "Mapa de valores faltantes"
    if "_box_outliers" in f:
        return "Outliers principales"
    if "_corr_heatmap" in f:
        return "Mapa de correlación"
    if "_likert_summary_scores" in f:
        return "Resumen Likert por ítem"
    if "_dimension_summary_scores" in f:
        return "Promedio por dimensión"
    if "_dimension_radar" in f:
        return "Radar de dimensiones"
    if "_likert_divergent" in f:
        return "Gráfico Likert divergente"
    if "_cronbach" in f:
        return "Consistencia interna del instrumento"
    if "_scree_plot" in f:
        return "Scree plot"
    if "_factor_loadings" in f:
        return "Mapa de cargas factoriales"
    if "_factor_model" in f:
        return "Diagrama del modelo factorial"

    m = re.search(r"_hist_(.+)\.png$", f)
    if m:
        return f"Distribución de {m.group(1).replace('_', ' ')}".title()

    m = re.search(r"_bar_(.+)\.png$", f)
    if m:
        return f"Frecuencia de categorías: {m.group(1).replace('_', ' ')}".title()

    m = re.search(r"_scatter_(.+)_vs_(.+)\.png$", f)
    if m:
        a = m.group(1).replace("_", " ")
        b = m.group(2).replace("_", " ")
        return f"Relación entre {a} y {b}".title()

    m = re.search(r"_likert_q(\d+)\.png$", f)
    if m:
        return f"Distribución de respuestas del ítem {m.group(1)}"

    return os.path.splitext(os.path.basename(f))[0].replace("_", " ").title()

def describe_plot(p: str) -> str:
    f = _fname(p)

    if "_missing" in f:
        return "Muestra la proporción de datos faltantes por variable y ayuda a evaluar la calidad general del dataset."
    if "_box_outliers" in f:
        return "Resume variables numéricas con valores extremos potenciales mediante diagramas de caja."
    if "_corr_heatmap" in f:
        return "Muestra la intensidad de relación entre variables numéricas."
    if "_scatter_" in f:
        return "Permite observar visualmente la asociación entre dos variables numéricas y posibles tendencias."
    if "_hist_" in f:
        return "Describe la forma de la distribución de una variable numérica, su dispersión y concentración."
    if "_bar_" in f:
        return "Resume la frecuencia de las principales categorías observadas en la variable."
    if "_cronbach" in f:
        return "Resume la consistencia interna del instrumento a partir del alfa de Cronbach."
    if "_likert_summary_scores" in f:
        return "Compara promedios entre dimensiones del instrumento en una escala común."
    if "_likert_divergent" in f:
        return "Visualiza la distribución de respuestas Likert desde desacuerdo hasta acuerdo."
    if "_dimension_radar" in f:
        return "Compara visualmente dimensiones del instrumento en un formato radial."
    if "_scree_plot" in f:
        return "Ayuda a identificar cuántos factores retener a partir de la varianza explicada."
    if "_factor_loadings" in f:
        return "Resume qué ítems cargan con mayor intensidad en cada factor."
    if "_factor_model" in f:
        return "Representa la estructura factorial inferida del instrumento."
    return "Gráfico generado automáticamente por SADI para apoyar la interpretación analítica del dataset."
    if clean == "dimension_summary_scores":
        return "Compara los promedios observados entre las dimensiones del instrumento."
    if clean == "dimension_radar":
        return "Resume visualmente el comportamiento relativo de las dimensiones del instrumento en formato radial."


def summarize_plot_tags(plots: list[str]) -> list[dict]:
    counts: dict[str, int] = {}
    for p in plots or []:
        tag = classify_plot_tag(p)
        counts[tag] = counts.get(tag, 0) + 1

    ordered = []
    priority = [
        "Calidad de datos",
        "Descriptivo",
        "Relacional",
        "Comparativo",
        "Psicométrico",
        "Factorial",
        "Analítico",
    ]
    for tag in priority:
        if tag in counts:
            ordered.append({"tag": tag, "count": counts[tag]})
    return ordered
def build_general_dataset_figure_catalog(
    *,
    dataset_id: int,
    plots_dir: str,
) -> list[dict]:
    catalog = []

    try:
        all_png = [p for p in os.listdir(plots_dir) if p.lower().endswith(".png")]
    except Exception:
        all_png = []

    ds_prefix = f"ds{dataset_id}_"
    names = [n for n in all_png if n.startswith(ds_prefix)]

    def order_key(n: str):
        low = n.lower()
        if "_missing" in low:
            return (0, n)
        if "_box" in low:
            return (1, n)
        if "_corr" in low or "_heatmap" in low:
            return (2, n)
        if "_scatter" in low:
            return (3, n)
        if "_hist" in low:
            return (4, n)
        if "_bar" in low:
            return (5, n)
        if "_likert" in low or "_cronbach" in low:
            return (6, n)
        if "_factor" in low or "_scree" in low:
            return (7, n)
        return (9, n)

    names.sort(key=order_key)

    for filename in names:
        catalog.append({
            "filename": f"plots/{filename}",
            "basename": filename,
            "title": prettify_plot_title(filename),
            "caption": describe_plot(filename),
            "tag": classify_plot_tag(filename),
        })

    return catalog