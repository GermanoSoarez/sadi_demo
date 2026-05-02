# blueprints/dataset/sadi_core.pyriate

from __future__ import annotations

from typing import Any

import pandas as pd

from .constans import normalize_dataset_type, normalize_research_area
from .analysis import suggest_research_area


def _unique(seq: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def build_sadi_core_plan(
    df: pd.DataFrame,
    *,
    dataset_type: str = "dataset",
    research_area: str = "general",
) -> dict[str, Any]:
    dataset_type = normalize_dataset_type(dataset_type)
    research_area = normalize_research_area(research_area)

    suggested_area = suggest_research_area(df)

    n_rows, n_cols = df.shape
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if c not in num_cols]

    recommended_analysis: list[str] = []
    recommended_plots: list[str] = []
    quick_recommendations: list[str] = []
    warnings: list[str] = []
    priority_order: list[str] = []
    narrative_focus = "general"

    # =========================
    # Base por tipo de dataset
    # =========================
    if dataset_type == "dataset":
        recommended_analysis += [
            "descriptive_stats",
            "missing_analysis",
            "duplicates_check",
            "categorical_summary",
            "numeric_summary",
        ]
        recommended_plots += [
            "missing_by_variable",
            "histograms",
            "boxplots",
            "top_categories",
        ]
        if len(num_cols) >= 2:
            recommended_analysis += ["correlation_analysis"]
            recommended_plots += ["correlation_heatmap", "top_scatter_pairs"]

        quick_recommendations += [
            "Revisar calidad de datos y missing antes de interpretar resultados.",
            "Priorizar variables numéricas con mayor variabilidad.",
        ]
        priority_order = ["missing", "box", "corr", "scatter", "hist", "bar"]

    elif dataset_type == "survey_normal":
        recommended_analysis += [
            "frequency_analysis",
            "categorical_summary",
            "response_distribution",
            "missing_analysis",
        ]
        recommended_plots += [
            "top_categories",
            "response_bars",
            "distribution_bars",
        ]
        quick_recommendations += [
            "Priorizar tablas de frecuencias y distribución de respuestas.",
            "Comparar categorías clave entre preguntas relevantes.",
        ]
        priority_order = ["bar", "hist", "missing", "corr", "scatter", "box"]

    elif dataset_type in {"survey_likert_5", "survey_likert_7"}:
        recommended_analysis += [
            "likert_summary",
            "item_distribution",
            "dimension_summary",
            "consistency_review",
        ]
        recommended_plots += [
            "likert_bars",
            "dimension_means",
            "item_ranking",
            "likert_heatmap",
        ]
        quick_recommendations += [
            "Calcular promedios por dimensión e ítem.",
            "Revisar consistencia interna antes de interpretar resultados globales.",
        ]
        priority_order = ["bar", "heatmap", "hist", "box", "corr", "scatter"]

    elif dataset_type == "multivariate":
        recommended_analysis += [
            "correlation_analysis",
            "pca",
            "clustering",
            "outlier_detection",
            "factor_analysis_if_applicable",
        ]
        recommended_plots += [
            "correlation_heatmap",
            "pca_plot",
            "cluster_plot",
            "outlier_boxplots",
        ]
        quick_recommendations += [
            "Evaluar reducción de dimensionalidad con PCA.",
            "Explorar agrupamientos y correlaciones altas.",
        ]
        priority_order = ["corr", "scatter", "box", "hist", "bar", "missing"]

    # =========================
    # Ajustes por área
    # =========================
    if research_area == "biomedicina":
        narrative_focus = "clinico"
        recommended_analysis += ["clinical_relationship_review", "extreme_values_review"]
        recommended_plots += ["clinical_boxplots", "clinical_scatter_pairs"]
        quick_recommendations += [
            "Buscar asociaciones clínicamente relevantes entre variables fisiológicas.",
            "Detectar valores extremos que puedan representar casos atípicos o errores.",
        ]
        priority_order = ["corr", "scatter", "box", "hist", "missing", "bar"]

    elif research_area == "educacion":
        narrative_focus = "educativo"
        recommended_analysis += ["performance_patterns", "group_comparison_if_available"]
        recommended_plots += ["performance_bars", "score_distributions"]
        quick_recommendations += [
            "Analizar rendimiento, asistencia y patrones académicos.",
            "Comparar resultados entre grupos o cursos si existen variables categóricas.",
        ]
        priority_order = ["bar", "hist", "corr", "box", "missing", "scatter"]

    elif research_area == "finanzas":
        narrative_focus = "financiero"
        recommended_analysis += ["variability_review", "outlier_detection", "strong_correlations_review"]
        recommended_plots += ["financial_boxplots", "financial_distributions"]
        quick_recommendations += [
            "Priorizar variabilidad, correlaciones y outliers financieros.",
            "Revisar relaciones entre precio, ingreso, ventas, costo y ganancia.",
        ]
        priority_order = ["box", "hist", "corr", "scatter", "missing", "bar"]

    elif research_area == "marketing":
        narrative_focus = "mercado"
        recommended_analysis += ["preference_patterns", "segment_hinting"]
        recommended_plots += ["category_preference_bars", "consumer_profile_bars"]
        quick_recommendations += [
            "Priorizar patrones de preferencia y comportamiento del cliente.",
            "Explorar segmentación por categorías y satisfacción.",
        ]
        priority_order = ["bar", "hist", "corr", "scatter", "box", "missing"]

    elif research_area == "agronomia":
        narrative_focus = "agronomico"
        recommended_analysis += ["environmental_correlations", "production_variability"]
        recommended_plots += ["environment_scatter_pairs", "production_boxplots"]
        quick_recommendations += [
            "Explorar relaciones entre clima, suelo y rendimiento.",
            "Revisar dispersión de variables productivas y ambientales.",
        ]
        priority_order = ["scatter", "corr", "box", "hist", "missing", "bar"]

    elif research_area == "social":
        narrative_focus = "social"
        recommended_analysis += ["sociodemographic_patterns"]
        recommended_plots += ["category_distribution_bars"]
        quick_recommendations += [
            "Priorizar variables demográficas y distribuciones categóricas.",
        ]
        priority_order = ["bar", "hist", "corr", "box", "missing", "scatter"]

    elif research_area == "ingenieria":
        narrative_focus = "ingenieria"
        recommended_analysis += ["performance_review", "process_variability"]
        recommended_plots += ["technical_scatter_pairs", "process_distributions"]
        quick_recommendations += [
            "Priorizar relaciones técnicas, rendimiento y variabilidad del sistema.",
        ]
        priority_order = ["corr", "scatter", "box", "hist", "missing", "bar"]

    elif research_area == "medio_ambiente":
        narrative_focus = "ambiental"
        recommended_analysis += ["environmental_trend_review", "extreme_values_review"]
        recommended_plots += ["environment_scatter_pairs", "environment_distributions"]
        quick_recommendations += [
            "Explorar relaciones ecológicas y valores extremos ambientales.",
        ]
        priority_order = ["scatter", "corr", "hist", "box", "missing", "bar"]

    elif research_area == "legal":
        narrative_focus = "juridico"
        recommended_analysis += ["case_distribution_review"]
        recommended_plots += ["legal_category_bars"]
        quick_recommendations += [
            "Priorizar distribución de casos y frecuencias categóricas.",
        ]
        priority_order = ["bar", "hist", "missing", "corr", "scatter", "box"]

    elif research_area == "psicologia":
        narrative_focus = "psicologico"
        recommended_analysis += ["scale_distribution_review", "consistency_review"]
        recommended_plots += ["score_distributions", "psychological_boxplots"]
        quick_recommendations += [
            "Revisar escalas, consistencia y dispersión de puntajes.",
        ]
        priority_order = ["hist", "box", "corr", "bar", "missing", "scatter"]

    if n_rows < 20:
        warnings.append("El dataset tiene pocas filas; algunos análisis pueden ser inestables.")
    if n_cols > 80:
        warnings.append("El dataset tiene muchas columnas; conviene priorizar variables clave.")

    return {
        "dataset_type": dataset_type,
        "research_area": research_area,
        "research_area_suggested": suggested_area,
        "recommended_analysis": _unique(recommended_analysis),
        "recommended_plots": _unique(recommended_plots),
        "quick_recommendations": _unique(quick_recommendations),
        "narrative_focus": narrative_focus,
        "warnings": _unique(warnings),
        "priority_order": priority_order,
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "n_num": len(num_cols),
        "n_cat": len(cat_cols),
    }