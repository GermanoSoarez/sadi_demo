from __future__ import annotations

import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PLOTS_DIR

from .constans import normalize_research_area, normalize_dataset_type
from blueprints.multivariate.services import run_rf_regression_analysis
from utils.plot_manager import normalize_plot_catalog, split_plots
def read_dataframe(path: str, delimiter: str | None = None) -> pd.DataFrame:
    import os
    import pandas as pd

    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"No existe el archivo: {path}")

    if os.path.isdir(path):
        raise ValueError(f"La ruta apunta a una carpeta, no a un archivo: {path}")

    if os.path.getsize(path) == 0:
        raise ValueError("El archivo está vacío.")

    ext = os.path.splitext(path)[1].lower()

    def _deduplicate_columns(columns):
        used = set()
        result = []

        for i, col in enumerate(columns, start=1):
            name = "" if col is None else str(col).strip()
            if not name or name.lower().startswith("unnamed:"):
                name = f"col_{i}"

            base = name
            n = 2
            while name in used:
                name = f"{base}_{n}"
                n += 1

            used.add(name)
            result.append(name)

        return result

    def _finalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        if df is None:
            raise ValueError("No se pudo leer el archivo como DataFrame.")

        df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
        df.columns = _deduplicate_columns(df.columns)
        df = df.reset_index(drop=True)
        return df

    if ext in {".xlsx", ".xls"}:
        try:
            df = pd.read_excel(path)
        except Exception as e:
            raise ValueError(f"No se pudo leer el archivo Excel: {e}") from e

        df = _finalize_dataframe(df)

        if df.empty and len(df.columns) == 0:
            raise ValueError("El archivo Excel no contiene datos utilizables.")

        return df

    if ext == ".csv":
        sep = delimiter or ","

        if sep == r"\t":
            sep = "\t"

        encodings = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]
        last_err = None

        for enc in encodings:
            try:
                df = pd.read_csv(
                path,
                sep=sep,
                encoding=enc,
                engine="python",
            )
                df = _finalize_dataframe(df)

                if len(df.columns) == 1:
                    only_col = str(df.columns[0])
                    suspicious_seps = [",", ";", "\t", "|", ":"]
                    if any(s in only_col for s in suspicious_seps):
                        raise ValueError(
                            f"El archivo parece tener un delimitador distinto a '{sep}'."
                        )

                if df.empty and len(df.columns) == 0:
                    raise ValueError("El CSV no contiene datos utilizables.")

                return df

            except Exception as e:
                last_err = e

        raise ValueError(f"No se pudo leer el CSV: {last_err}") from last_err

    if ext == ".tsv":
        try:
            df = pd.read_csv(
                path,
                sep="\t",
                encoding="utf-8-sig",
                engine="python",
            )
        except Exception:
            try:
                df = pd.read_csv(
                    path,
                    sep="\t",
                    encoding="cp1252",
                    engine="python",
                )
            except Exception as e:
                raise ValueError(f"No se pudo leer el TSV: {e}") from e

        df = _finalize_dataframe(df)

        if df.empty and len(df.columns) == 0:
            raise ValueError("El TSV no contiene datos utilizables.")

        return df

    if ext == ".json":
        try:
            df = pd.read_json(path)
        except Exception as e:
            raise ValueError(f"No se pudo leer el JSON: {e}") from e

        if not isinstance(df, pd.DataFrame):
            raise ValueError("El JSON no produjo una tabla válida.")

        df = _finalize_dataframe(df)

        if df.empty and len(df.columns) == 0:
            raise ValueError("El JSON no contiene datos utilizables.")

        return df

    raise ValueError(f"Formato no soportado: {ext}")

def suggest_research_area(df: pd.DataFrame) -> str:
    """
    Sugiere el área de investigación a partir de los nombres de columnas.
    Usa las áreas actuales del sistema SADI.
    """
    try:
        cols = [str(c).strip().lower() for c in df.columns]
    except Exception:
        return "general"

    if not cols:
        return "general"

    biomedicina_terms = {
        "paciente", "diagnostico", "diagnóstico", "tratamiento", "presion", "presión",
        "imc", "glucosa", "colesterol", "peso", "talla", "sexo", "edad",
        "hospital", "enfermedad", "medico", "médico", "salud", "sintoma", "síntoma",
        "frecuencia_cardiaca", "temperatura", "hemoglobina", "clinico", "clínico"
    }

    finanzas_terms = {
        "ingreso", "gasto", "costo", "coste", "ventas", "balance", "capital",
        "utilidad", "beneficio", "rentabilidad", "activo", "pasivo", "flujo",
        "caja", "interes", "interés", "credito", "crédito", "deuda", "precio",
        "facturacion", "facturación", "ahorro", "inversion", "inversión"
    }

    educacion_terms = {
        "alumno", "alumnos", "estudiante", "estudiantes", "curso", "materia",
        "nota", "nota_final", "calificacion", "calificación", "asistencia",
        "docente", "profesor", "aprendizaje", "rendimiento", "semestre", "carrera",
        "evaluacion", "evaluación", "promedio", "facultad"
    }

    marketing_terms = {
        "cliente", "clientes", "producto", "marca", "satisfaccion", "satisfacción",
        "compra", "consumo", "segmento", "mercado", "campaña", "campana",
        "publicidad", "fidelidad", "preferencia", "intencion_compra", "intención_compra"
    }

    agronomia_terms = {
        "suelo", "cultivo", "parcela", "produccion", "producción", "riego",
        "fertilizante", "siembra", "cosecha", "agricola", "agrícola", "temperatura",
        "humedad", "ph", "nitrógeno", "nitrogeno", "potasio", "maiz", "maíz"
    }

    social_terms = {
        "hogar", "familia", "comunidad", "encuestado", "encuestada", "ocupacion",
        "ocupación", "nivel_social", "ingreso_familiar", "poblacion", "población",
        "barrio", "vivienda", "estado_civil", "participacion", "participación"
    }

    ingenieria_terms = {
        "proceso", "sensor", "voltaje", "corriente", "temperatura", "presion",
        "presión", "eficiencia", "produccion", "producción", "sistema",
        "latencia", "rendimiento", "error", "tiempo_respuesta", "algoritmo",
        "servidor", "cpu", "memoria"
    }

    medio_ambiente_terms = {
        "co2", "temperatura", "humedad", "precipitacion", "precipitación", "rio",
        "río", "bosque", "contaminacion", "contaminación", "aire", "agua",
        "emisiones", "residuo", "residuos", "clima", "ecosistema"
    }

    legal_terms = {
        "delito", "demanda", "caso", "sentencia", "expediente", "norma",
        "articulo", "artículo", "juzgado", "tribunal", "pena", "ley",
        "resolucion", "resolución", "fiscalia", "fiscalía"
    }

    psicologia_terms = {
        "ansiedad", "estres", "estrés", "depresion", "depresión", "conducta",
        "emocion", "emoción", "autoestima", "personalidad", "bienestar",
        "motivacion", "motivación", "escala", "ítem", "item", "test"
    }

    def score(term_set: set[str]) -> int:
        total = 0
        for c in cols:
            for t in term_set:
                if t in c:
                    total += 1
        return total

    scores = {
        "biomedicina": score(biomedicina_terms),
        "finanzas": score(finanzas_terms),
        "educacion": score(educacion_terms),
        "marketing": score(marketing_terms),
        "agronomia": score(agronomia_terms),
        "social": score(social_terms),
        "ingenieria": score(ingenieria_terms),
        "medio_ambiente": score(medio_ambiente_terms),
        "legal": score(legal_terms),
        "psicologia": score(psicologia_terms),
    }

    best_area = max(scores, key=scores.get)
    best_score = scores[best_area]

    if best_score <= 0:
        return "general"

    return best_area


def detect_target_variable(df: pd.DataFrame) -> dict:
    import pandas as pd

    result = {
        "target_candidate": None,
        "target_type": None,   # regression | classification | None
        "target_reason": None,
        "binary_target_candidates": [],
        "ranked_candidates": [],
    }

    if df is None or df.empty:
        return result

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    cat_cols = [c for c in df.columns if c not in num_cols and c not in dt_cols]

    target_keywords_strong = [
        "target", "label", "class", "resultado", "outcome",
        "objetivo", "response", "diagnostico", "diagnóstico",
        "aprobado", "estado", "riesgo", "score"
    ]

    target_keywords_soft = [
        "yield", "venta", "ventas", "price", "precio",
        "ingreso", "rent", "profit", "ganancia", "demand",
        "demanda", "performance", "rendimiento"
    ]

    id_keywords = [
        "id", "codigo", "código", "code", "folio",
        "nro", "número", "numero", "index", "idx"
    ]

    def is_id_like(col_name: str, series: pd.Series) -> bool:
        low = str(col_name).strip().lower()
        if any(k == low or low.endswith("_" + k) or low.startswith(k + "_") for k in id_keywords):
            return True
        s = series.dropna()
        if s.empty:
            return False
        # muchos únicos y nombre tipo id
        if s.nunique() >= max(10, int(len(s) * 0.9)) and any(k in low for k in id_keywords):
            return True
        return False

    candidates = []

    # 1) Candidatos categóricos
    for c in cat_cols:
        s = df[c].dropna()
        if s.empty:
            continue

        nun = s.nunique()
        if nun < 2:
            continue
        if nun > 30:
            continue
        if is_id_like(c, s):
            continue

        low = str(c).strip().lower()
        score = 0.0
        reason_parts = []

        if any(k in low for k in target_keywords_strong):
            score += 10
            reason_parts.append("nombre altamente compatible con variable objetivo")
        elif any(k in low for k in target_keywords_soft):
            score += 5
            reason_parts.append("nombre parcialmente compatible con variable objetivo")

        if nun == 2:
            score += 6
            reason_parts.append("variable binaria apta para clasificación")
            result["binary_target_candidates"].append(c)
        elif 3 <= nun <= 10:
            score += 3
            reason_parts.append("número de clases razonable para clasificación")

        missing_pct = float(df[c].isna().mean() * 100)
        score += max(0, 3 - (missing_pct / 10))

        candidates.append({
            "column": c,
            "type": "classification",
            "score": round(score, 4),
            "reason": ", ".join(reason_parts) if reason_parts else "candidata categórica razonable",
        })

    # 2) Candidatos numéricos
    for c in num_cols:
        s = df[c].dropna()
        if s.empty:
            continue
        if s.nunique() <= 1:
            continue
        if is_id_like(c, s):
            continue

        low = str(c).strip().lower()
        score = 0.0
        reason_parts = []

        if any(k in low for k in target_keywords_strong):
            score += 10
            reason_parts.append("nombre altamente compatible con variable objetivo")
        elif any(k in low for k in target_keywords_soft):
            score += 5
            reason_parts.append("nombre parcialmente compatible con variable objetivo")

        std = float(s.std()) if s.shape[0] > 1 else 0.0
        rng = float(s.max() - s.min()) if s.shape[0] > 0 else 0.0
        missing_pct = float(df[c].isna().mean() * 100)

        if std > 0:
            score += min(std, 5)
            reason_parts.append("variabilidad útil")
        if rng > 0:
            score += min(rng / 100.0, 3)

        score += max(0, 3 - (missing_pct / 10))

        # penalizar columnas con demasiados únicos si parecen identificadores puros
        uniq_ratio = s.nunique() / max(len(s), 1)
        if uniq_ratio > 0.95 and "score" not in low and "yield" not in low and "price" not in low:
            score -= 2

        candidates.append({
            "column": c,
            "type": "regression",
            "score": round(score, 4),
            "reason": ", ".join(reason_parts) if reason_parts else "candidata numérica razonable",
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    result["ranked_candidates"] = candidates[:10]

    if candidates:
        top = candidates[0]
        result["target_candidate"] = top["column"]
        result["target_type"] = top["type"]
        result["target_reason"] = f"SADI seleccionó '{top['column']}' como variable objetivo porque presenta {top['reason']}."

    return result

def summarize_column(s: pd.Series) -> dict:
    name = str(getattr(s, "name", "") or "")
    n_total = int(len(s))
    n_missing = int(s.isna().sum())
    n_valid = int(n_total - n_missing)
    missing_pct = (n_missing / n_total * 100.0) if n_total else 0.0

    if s.dtype == "object":
        s_try = pd.to_numeric(s, errors="coerce")
        non_null = s.notna().sum()
        conv = s_try.notna().sum()
        if non_null and (conv / non_null) >= 0.80:
            s = s_try

    dtype_raw = str(s.dtype)
    is_bool = pd.api.types.is_bool_dtype(s)
    is_num = pd.api.types.is_numeric_dtype(s) and not is_bool
    is_dt = pd.api.types.is_datetime64_any_dtype(s)

    out = {
        "name": name,
        "dtype": "numeric" if is_num else ("datetime" if is_dt else ("boolean" if is_bool else "categorical")),
        "dtype_raw": dtype_raw,
        "n_total": n_total,
        "n_valid": n_valid,
        "n_missing": n_missing,
        "missing_pct": float(missing_pct),
        "numeric": {},
        "categorical": {},
        "top_categories": [],
    }

    if n_valid == 0:
        return out

    if is_num:
        vals = pd.to_numeric(s, errors="coerce").dropna()
        if len(vals) == 0:
            return out
        out["numeric"] = {
            "mean": float(vals.mean()),
            "median": float(vals.median()),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else None,
            "q1": round(float(vals.quantile(0.25)), 4) if len(vals) else None,
            "q3": round(float(vals.quantile(0.75)), 4) if len(vals) else None,
        }
        return out

    if is_dt:
        vals = pd.to_datetime(s, errors="coerce").dropna()
        if len(vals) == 0:
            return out
        out["categorical"] = {
            "n_unique": int(vals.nunique()),
            "top": str(vals.value_counts().index[0]),
            "top_freq": int(vals.value_counts().iloc[0]),
        }
        return out

    s2 = s.where(s.notna(), np.nan)
    if is_bool:
        vc = s2.value_counts(dropna=True)
    else:
        vc = s2.astype(str).replace("nan", np.nan).dropna().value_counts()

    n_unique = int(vc.size)
    top = str(vc.index[0]) if n_unique else None
    top_freq = int(vc.iloc[0]) if n_unique else None

    out["categorical"] = {
        "n_unique": n_unique,
        "top": top,
        "top_freq": top_freq,
    }

    top_categories = []
    if n_valid:
        topn = vc.head(10)
        for k, cnt in topn.items():
            pct = (int(cnt) / n_valid * 100.0) if n_valid else 0.0
            top_categories.append({"category": str(k), "count": int(cnt), "pct": float(pct)})
    out["top_categories"] = top_categories

    return out


def analyze_general_dataset(df: pd.DataFrame) -> dict:
    import pandas as pd

    meta = {}

    n_rows, n_cols = int(df.shape[0]), int(df.shape[1])
    meta["n_rows"] = n_rows
    meta["n_cols"] = n_cols

    total_cells = n_rows * n_cols
    missing_total = int(df.isna().sum().sum()) if total_cells else 0
    meta["missing_global_pct"] = round((missing_total / total_cells) * 100, 2) if total_cells else 0.0

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    cat_cols = [c for c in df.columns if c not in num_cols and c not in dt_cols]

    meta["num_cols"] = num_cols
    meta["cat_cols"] = cat_cols
    meta["dt_cols"] = dt_cols
    meta["n_num"] = len(num_cols)
    meta["n_cat"] = len(cat_cols)
    meta["n_dt"] = len(dt_cols)

    # =========================
    # Correlaciones altas
    # =========================
    high_corr_pairs = []
    if len(num_cols) >= 2:
        try:
            corr = df[num_cols].corr(numeric_only=True)
            for i in range(len(corr.columns)):
                for j in range(i + 1, len(corr.columns)):
                    val = corr.iloc[i, j]
                    if pd.notna(val) and abs(val) >= 0.70:
                        high_corr_pairs.append({
                            "col1": corr.columns[i],
                            "col2": corr.columns[j],
                            "corr": round(float(val), 4),
                        })
        except Exception:
            pass

    meta["high_corr_pairs"] = high_corr_pairs[:12]

    # =========================
    # Variables más variables
    # =========================
    top_numeric_by_variability = []
    for c in num_cols:
        s = df[c].dropna()
        if s.empty:
            continue

        std = float(s.std()) if s.shape[0] > 1 else 0.0
        rng = float(s.max() - s.min()) if s.shape[0] > 0 else 0.0

        top_numeric_by_variability.append({
            "column": c,
            "std": round(std, 4),
            "range": round(rng, 4),
        })

    top_numeric_by_variability.sort(key=lambda x: x["std"], reverse=True)
    meta["top_numeric_by_variability"] = top_numeric_by_variability[:8]

    # =========================
    # Ranking de variables clave
    # =========================
    variable_importance = []
    for c in num_cols:
        s = df[c].dropna()
        if s.empty:
            continue

        std = float(s.std()) if s.shape[0] > 1 else 0.0
        missing_pct = round(df[c].isna().mean() * 100, 2)

        score = std * (1 - (missing_pct / 100))

        variable_importance.append({
            "column": c,
            "score": round(score, 4),
            "std": round(std, 4),
            "missing_pct": missing_pct,
        })

    variable_importance.sort(key=lambda x: x["score"], reverse=True)
    meta["variable_importance"] = variable_importance[:10]

    # =========================
    # Outliers por IQR
    # =========================
    outlier_summary = []
    for c in num_cols:
        s = df[c].dropna()
        if s.shape[0] < 8:
            continue

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue

        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        n_out = int(((s < low) | (s > high)).sum())
        pct_out = round((n_out / len(s)) * 100, 2) if len(s) else 0.0

        if n_out > 0:
            outlier_summary.append({
                "column": c,
                "n_outliers": n_out,
                "pct_outliers": pct_out,
            })

    outlier_summary.sort(key=lambda x: x["n_outliers"], reverse=True)
    meta["outlier_summary"] = outlier_summary[:8]

    # =========================
    # Patrones rápidos
    # =========================
    pattern_notes = []

    if len(meta["high_corr_pairs"]) >= 3:
        pattern_notes.append(
            "Se observan múltiples relaciones fuertes entre variables numéricas, lo que sugiere estructura interna relevante."
        )

    if len(meta["outlier_summary"]) >= 2:
        pattern_notes.append(
            "Se detectan valores atípicos en varias variables, por lo que conviene revisar robustez y limpieza antes de modelar."
        )

    if len(meta["top_numeric_by_variability"]) >= 1:
        pattern_notes.append(
            f"La variable con mayor dispersión observada fue '{meta['top_numeric_by_variability'][0]['column']}'."
        )

    if meta["missing_global_pct"] >= 10:
        pattern_notes.append(
            "El porcentaje de datos faltantes podría afectar la estabilidad de análisis más avanzados."
        )

    meta["pattern_notes"] = pattern_notes

    # =========================
    # Detección automática de target
    # =========================
    target_info = detect_target_variable(df)

    meta["target_candidate"] = target_info.get("target_candidate")
    meta["target_type"] = target_info.get("target_type")
    meta["target_reason"] = target_info.get("target_reason")
    meta["binary_target_candidates"] = target_info.get("binary_target_candidates", [])
    meta["ranked_target_candidates"] = target_info.get("ranked_candidates", [])

    return meta

def suggest_dataset_analysis_plan(
    df: pd.DataFrame,
    *,
    dataset_type: str = "dataset",
    research_area: str = "general",
) -> dict:
    """
    Sugiere análisis, gráficos y enfoque interpretativo
    según área de investigación + tipo de dataset.
    Pensado para el blueprint general 'dataset'.
    """
    dataset_type = normalize_dataset_type(dataset_type)
    research_area = normalize_research_area(research_area)

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    n_rows, n_cols = df.shape

    recommended_analysis = [
        "descriptive_stats",
        "missing_analysis",
        "duplicates_check",
    ]

    recommended_plots = [
        "missing_by_variable",
    ]

    narrative_focus = "general"
    warnings = []

    if len(num_cols) >= 1:
        recommended_analysis.append("numeric_summary")
        recommended_plots.append("histograms")
        recommended_plots.append("boxplots")

    if len(cat_cols) >= 1:
        recommended_analysis.append("categorical_summary")
        recommended_plots.append("top_categories")

    if len(num_cols) >= 2:
        recommended_analysis.append("correlation_analysis")
        recommended_plots.append("correlation_heatmap")
        recommended_plots.append("top_scatter_pairs")

    if n_rows < 20:
        warnings.append("El dataset tiene pocas filas; algunos análisis pueden ser inestables.")
    if n_cols > 80:
        warnings.append("El dataset tiene muchas columnas; conviene priorizar variables clave.")

    # =========================================
    # Ajustes por área de investigación
    # =========================================
    if research_area == "biomedicina":
        narrative_focus = "clinico"
        recommended_analysis += ["outlier_detection", "strong_correlations_review"]
        recommended_plots += ["clinical_distributions", "clinical_boxplots"]

    elif research_area == "educacion":
        narrative_focus = "educativo"
        recommended_analysis += ["performance_patterns", "group_comparison_if_available"]
        recommended_plots += ["performance_bars", "score_distributions"]

    elif research_area == "finanzas":
        narrative_focus = "financiero"
        recommended_analysis += ["outlier_detection", "variance_review", "strong_correlations_review"]
        recommended_plots += ["financial_distributions", "financial_boxplots"]

    elif research_area == "marketing":
        narrative_focus = "mercado"
        recommended_analysis += ["segment_hinting", "preference_patterns"]
        recommended_plots += ["category_preference_bars", "consumer_profiles"]

    elif research_area == "agronomia":
        narrative_focus = "agronomico"
        recommended_analysis += ["environmental_correlations", "production_variability"]
        recommended_plots += ["environmental_boxplots", "production_scatter_pairs"]

    elif research_area == "social":
        narrative_focus = "social"
        recommended_analysis += ["sociodemographic_patterns", "distribution_review"]
        recommended_plots += ["category_distribution_bars"]

    elif research_area == "ingenieria":
        narrative_focus = "ingenieria"
        recommended_analysis += ["process_variability", "performance_review", "strong_correlations_review"]
        recommended_plots += ["process_histograms", "technical_scatter_pairs"]

    elif research_area == "medio_ambiente":
        narrative_focus = "ambiental"
        recommended_analysis += ["environmental_correlations", "extreme_values_review"]
        recommended_plots += ["environmental_trends_if_possible", "environmental_boxplots"]

    elif research_area == "legal":
        narrative_focus = "juridico"
        recommended_analysis += ["frequency_patterns", "case_distribution_review"]
        recommended_plots += ["legal_category_bars"]

    elif research_area == "psicologia":
        narrative_focus = "psicologico"
        recommended_analysis += ["scale_distribution_review", "variability_review"]
        recommended_plots += ["psychological_distributions", "score_boxplots"]

    else:
        narrative_focus = "general"

    # limpiar duplicados preservando orden
    def _unique(seq):
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    recommended_analysis = _unique(recommended_analysis)
    recommended_plots = _unique(recommended_plots)

    return {
        "dataset_type": dataset_type,
        "research_area": research_area,
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "n_num": len(num_cols),
        "n_cat": len(cat_cols),
        "recommended_analysis": recommended_analysis,
        "recommended_plots": recommended_plots,
        "narrative_focus": narrative_focus,
        "warnings": warnings,
    }


def _annot_box(ax, text: str):
    ax.text(
        0.99, 0.98, text,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.9),
    )


def _fmt_num(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.3g}"
    return str(x)

def auto_select_best_regression_model(
    *,
    df,
    dataset_id: int,
    plots_dir: str,
    target_col: str,
    manifest_data: dict | None = None,
    ) -> dict:
    """
    Ejecuta (si hace falta) regresión lineal y random forest,
    compara resultados y devuelve el mejor modelo sugerido por SADI.
    """
    manifest_data = manifest_data or {}
    if not isinstance(manifest_data, dict):
        manifest_data = {}

    model_results = manifest_data.get("model_results", {})
    if not isinstance(model_results, dict):
        model_results = {}

    regression_result = model_results.get("regression_result")
    rf_result = model_results.get("rf_result")

    # Ejecutar si no existen
    if not regression_result:
        try:
            regression_result = run_regression_analysis(
                df=df,
                dataset_id=dataset_id,
                plots_dir=plots_dir,
                target_col=target_col,
            )
            model_results["regression_result"] = regression_result
        except Exception:
            regression_result = None

    if not rf_result:
        try:
            rf_result = run_rf_regression_analysis(
                df=df,
                dataset_id=dataset_id,
                plots_dir=plots_dir,
                target_col=target_col,
            )
            model_results["rf_result"] = rf_result
        except Exception:
            rf_result = None

    manifest_data["model_results"] = model_results

    comparison = build_model_comparison_summary(
        regression_result=regression_result,
        rf_result=rf_result,
    )

    best_model = comparison.get("winner")
    best_reason = comparison.get("recommendation")

    manifest_data["best_model_selection"] = {
        "target_col": target_col,
        "problem_type": "regression",
        "best_model": best_model,
        "reason": best_reason,
        "comparison_summary": comparison.get("summary", []),
    }

    return manifest_data

def _safe_slug(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "_", text)
    return text[:40]


def generate_general_dataset_plots(
    df: pd.DataFrame,
    dataset_id: int,
    *,
    dataset_type: str = "dataset",
    research_area: str = "general",
) -> list[str]:
    plots: list[str] = []
    n = len(df)

    dataset_type = normalize_dataset_type(dataset_type)
    research_area = normalize_research_area(research_area)

    num_all: list[str] = []
    num_series: dict[str, pd.Series] = {}

    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= max(10, int(0.30 * n)):
            num_all.append(c)
            num_series[c] = s

    cat_all: list[str] = []
    for c in df.columns:
        if c in num_all:
            continue
        is_cat = (
            df[c].dtype == "object"
            or isinstance(df[c].dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(df[c])
        )
        if is_cat:
            cat_all.append(c)

    num_cols = num_all[:6]
    cat_cols = cat_all[:6]

    try:
        miss_pct = (df.isna().mean() * 100.0).sort_values(ascending=False)
        miss_pct = miss_pct[miss_pct > 0].head(20)

        if len(miss_pct) > 0:
            fig, ax = plt.subplots(figsize=(8, 5))
            y = list(range(len(miss_pct)))
            ax.barh(y, miss_pct.values)
            ax.set_yticks(y)
            ax.set_yticklabels([str(x) for x in miss_pct.index])
            ax.invert_yaxis()
            ax.set_title("Missing por variable (Top 20)")
            ax.set_xlabel("% Missing")

            for i, v in enumerate(miss_pct.values):
                ax.text(v + 0.2, i, f"{v:.1f}%", va="center", fontsize=9)

            plt.tight_layout()
            fname = f"ds{dataset_id}_missing.png"
            outpath = os.path.join(PLOTS_DIR, fname)
            plt.savefig(outpath, dpi=140)
            plt.close()
            plots.append(f"plots/{fname}")
    except Exception:
        pass

    for col in num_cols:
        try:
            series = num_series.get(col)
            if series is None:
                series = pd.to_numeric(df[col], errors="coerce")

            n_total = len(df[col])
            n_valid = int(series.notna().sum())
            n_missing = n_total - n_valid
            miss_pct = (n_missing / n_total * 100.0) if n_total else 0.0

            series = series.dropna()
            if series.empty:
                continue

            mean = float(series.mean())
            median = float(series.median())
            std = float(series.std(ddof=1)) if len(series) > 1 else float("nan")
            mn = float(series.min())
            mx = float(series.max())

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(series.values, bins=20)
            ax.set_title(f"Distribución: {col}")
            ax.set_xlabel(str(col))
            ax.set_ylabel("Frecuencia")

            info = (
                f"n={n_valid}/{n_total}\n"
                f"missing={miss_pct:.1f}%\n"
                f"mean={_fmt_num(mean)} med={_fmt_num(median)}\n"
                f"std={_fmt_num(std)}\n"
                f"min={_fmt_num(mn)} max={_fmt_num(mx)}"
            )
            _annot_box(ax, info)

            plt.tight_layout()
            fname = f"ds{dataset_id}_hist_{_safe_slug(col)}.png"
            outpath = os.path.join(PLOTS_DIR, fname)
            plt.savefig(outpath, dpi=140)
            plt.close()
            plots.append(f"plots/{fname}")
        except Exception:
            continue

    try:
        variances = []
        for c in num_all:
            s = num_series.get(c)
            if s is None:
                s = pd.to_numeric(df[c], errors="coerce")
            s = s.dropna()
            if len(s) >= 5:
                variances.append((c, float(s.var())))

        variances.sort(key=lambda x: x[1], reverse=True)
        box_cols = [c for c, _ in variances[:4]]

        data = []
        labels = []
        stats_txt = []

        for c in box_cols:
            s = num_series[c].dropna()
            if len(s) == 0:
                continue

            data.append(s.values)
            labels.append(str(c)[:12])

            q1 = float(s.quantile(0.25))
            q2 = float(s.quantile(0.50))
            q3 = float(s.quantile(0.75))
            iqr = q3 - q1
            lo = q1 - 1.5 * iqr
            hi = q3 + 1.5 * iqr
            outliers = int(((s < lo) | (s > hi)).sum())

            stats_txt.append(f"{c}: Q1={_fmt_num(q1)} Med={_fmt_num(q2)} Q3={_fmt_num(q3)} out={outliers}")

        if data:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.boxplot(data, vert=True, tick_labels=labels, showfliers=True)
            ax.set_title("Boxplot (Outliers) — Top variables numéricas")
            ax.set_ylabel("Valor")
            _annot_box(ax, "\n".join(stats_txt[:4]))

            plt.tight_layout()
            fname = f"ds{dataset_id}_box_outliers.png"
            outpath = os.path.join(PLOTS_DIR, fname)
            plt.savefig(outpath, dpi=140)
            plt.close()
            plots.append(f"plots/{fname}")
    except Exception:
        pass

    for col in cat_cols:
        try:
            vc = df[col].dropna().astype(str).value_counts().head(10)
            if vc.empty:
                continue

            plt.figure(figsize=(7, 4))
            plt.bar(range(len(vc)), vc.values)
            plt.xticks(range(len(vc)), vc.index, rotation=30, ha="right")
            plt.title(f"Top categorías: {col}")
            plt.xlabel(str(col))
            plt.ylabel("Frecuencia")
            plt.tight_layout()

            fname = f"ds{dataset_id}_bar_{_safe_slug(col)}.png"
            outpath = os.path.join(PLOTS_DIR, fname)
            plt.savefig(outpath, dpi=140)
            plt.close()
            plots.append(f"plots/{fname}")
        except Exception:
            continue

    try:
        if len(num_all) >= 2:
            use = num_all[:12]
            num_df = pd.DataFrame({c: num_series[c] for c in use})
            corr = num_df.corr()

            if corr.shape[0] >= 2:
                plt.figure(figsize=(8, 6))
                plt.imshow(corr.values, aspect="auto")
                plt.colorbar()
                plt.xticks(range(len(use)), [str(c)[:10] for c in use], rotation=45, ha="right")
                plt.yticks(range(len(use)), [str(c)[:10] for c in use])
                plt.title("Mapa de correlación (numéricas)")
                plt.tight_layout()

                fname = f"ds{dataset_id}_corr_heatmap.png"
                outpath = os.path.join(PLOTS_DIR, fname)
                plt.savefig(outpath, dpi=140)
                plt.close()
                plots.append(f"plots/{fname}")

                pairs = []
                for i, a in enumerate(use):
                    for b in use[i + 1:]:
                        val = corr.loc[a, b]
                        if pd.notna(val):
                            pairs.append((a, b, float(val)))

                pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                top_pairs = [p for p in pairs if abs(p[2]) >= 0.70][:6]

                for a, b, r in top_pairs:
                    x = num_series[a]
                    y = num_series[b]
                    m = x.notna() & y.notna()
                    nn = int(m.sum())

                    if nn < 10:
                        continue

                    xv = x[m].values
                    yv = y[m].values

                    fig, ax = plt.subplots(figsize=(7, 5))
                    ax.scatter(xv, yv, s=12, alpha=0.7)
                    ax.set_title(f"Scatter: {a} vs {b}")
                    ax.set_xlabel(str(a))
                    ax.set_ylabel(str(b))

                    try:
                        coef = np.polyfit(xv, yv, 1)
                        xs = np.linspace(np.min(xv), np.max(xv), 60)
                        ys = coef[0] * xs + coef[1]
                        ax.plot(xs, ys, linewidth=2)
                    except Exception:
                        pass

                    info = f"r={r:.3f}\nN={nn}"
                    _annot_box(ax, info)

                    plt.tight_layout()
                    fname = f"ds{dataset_id}_scatter_{_safe_slug(a)}_vs_{_safe_slug(b)}.png"
                    outpath = os.path.join(PLOTS_DIR, fname)
                    plt.savefig(outpath, dpi=140)
                    plt.close()
                    plots.append(f"plots/{fname}")
    except Exception:
        pass

    # =========================================================
    # ORDEN INTELIGENTE SADI SEGÚN ÁREA + TIPO
    # =========================================================
    def _plot_kind(p: str) -> str:
        low = str(p).lower()
        if "_missing" in low:
            return "missing"
        if "_box_" in low or "_box." in low or "_box" in low:
            return "box"
        if "_corr_" in low or "_corr." in low or "_corr" in low:
            return "corr"
        if "_scatter_" in low or "_scatter" in low:
            return "scatter"
        if "_hist_" in low or "_hist" in low:
            return "hist"
        if "_bar_" in low or "_bar" in low:
            return "bar"
        return "other"

    def _priority_for_area(area: str, dtype: str) -> dict[str, int]:
        # prioridad menor = aparece antes
        if dtype != "dataset":
            return {
                "missing": 0,
                "box": 1,
                "corr": 2,
                "scatter": 3,
                "hist": 4,
                "bar": 5,
                "other": 9,
            }

        area_map = {
            "biomedicina": {
                "missing": 0, "box": 1, "corr": 2, "scatter": 3, "hist": 4, "bar": 5, "other": 9
            },
            "educacion": {
                "missing": 0, "bar": 1, "hist": 2, "box": 3, "corr": 4, "scatter": 5, "other": 9
            },
            "finanzas": {
                "missing": 0, "box": 1, "hist": 2, "corr": 3, "scatter": 4, "bar": 5, "other": 9
            },
            "marketing": {
                "missing": 0, "bar": 1, "hist": 2, "corr": 3, "scatter": 4, "box": 5, "other": 9
            },
            "agronomia": {
                "missing": 0, "scatter": 1, "corr": 2, "box": 3, "hist": 4, "bar": 5, "other": 9
            },
            "social": {
                "missing": 0, "bar": 1, "hist": 2, "corr": 3, "box": 4, "scatter": 5, "other": 9
            },
            "ingenieria": {
                "missing": 0, "corr": 1, "scatter": 2, "box": 3, "hist": 4, "bar": 5, "other": 9
            },
            "medio_ambiente": {
                "missing": 0, "scatter": 1, "corr": 2, "box": 3, "hist": 4, "bar": 5, "other": 9
            },
            "legal": {
                "missing": 0, "bar": 1, "hist": 2, "corr": 3, "box": 4, "scatter": 5, "other": 9
            },
            "psicologia": {
                "missing": 0, "hist": 1, "box": 2, "corr": 3, "bar": 4, "scatter": 5, "other": 9
            },
            "general": {
                "missing": 0, "box": 1, "corr": 2, "scatter": 3, "hist": 4, "bar": 5, "other": 9
            },
        }
        return area_map.get(area, area_map["general"])

    priority_map = _priority_for_area(research_area, dataset_type)

    plots = sorted(
        plots,
        key=lambda p: (
            priority_map.get(_plot_kind(p), 99),
            _plot_kind(p),
            p,
        )
    )

    return plots
def interpret_stat(value: float, metric: str, research_area: str) -> str:
    """
    Genera interpretación automática según área de investigación.
    """

    research_area = normalize_research_area(research_area)

    if abs(value) >= 0.7:
        level = "alta"
    elif abs(value) >= 0.4:
        level = "moderada"
    elif abs(value) >= 0.2:
        level = "baja"
    else:
        level = "muy baja"

    if research_area == "biomedicina":
        return (
            f"Se observa una asociación {level} ({value:.2f}), "
            "lo que podría indicar una relación clínicamente relevante entre las variables analizadas."
        )

    elif research_area == "finanzas":
        return (
            f"Existe una relación {level} ({value:.2f}), "
            "lo que sugiere un posible impacto importante en el comportamiento financiero observado."
        )

    elif research_area == "educacion":
        return (
            f"Se evidencia una relación {level} ({value:.2f}), "
            "que podría influir en el rendimiento o comportamiento académico de la población evaluada."
        )

    elif research_area == "marketing":
        return (
            f"Se detecta una relación {level} ({value:.2f}), "
            "que podría estar asociada con preferencias, segmentación o comportamiento del consumidor."
        )

    elif research_area == "agronomia":
        return (
            f"Se observa una relación {level} ({value:.2f}), "
            "que podría influir en variables agronómicas como producción, suelo o condiciones ambientales."
        )

    elif research_area == "social":
        return (
            f"Se observa una relación {level} ({value:.2f}), "
            "que podría reflejar dinámicas sociales relevantes dentro de la población analizada."
        )

    elif research_area == "ingenieria":
        return (
            f"Se detecta una relación {level} ({value:.2f}), "
            "lo que podría ser relevante para el rendimiento, control o eficiencia del sistema estudiado."
        )

    elif research_area == "medio_ambiente":
        return (
            f"Se observa una relación {level} ({value:.2f}), "
            "que podría estar vinculada con patrones ambientales o variaciones ecológicas relevantes."
        )

    elif research_area == "legal":
        return (
            f"Se detecta una relación {level} ({value:.2f}), "
            "que podría ayudar a identificar patrones jurídicos o de distribución de casos."
        )

    elif research_area == "psicologia":
        return (
            f"Se evidencia una relación {level} ({value:.2f}), "
            "que podría ser relevante para la interpretación de variables psicológicas o conductuales."
        )

    else:
        return f"Se observa una relación {level} entre las variables ({value:.2f})."
    
def analyze_dataset_with_recommendations(
    df: pd.DataFrame,
    *,
    dataset_type: str = "dataset",
    research_area: str = "general",
) -> dict:
    from .sadi_core import build_sadi_core_plan

    dataset_type = normalize_dataset_type(dataset_type)
    research_area = normalize_research_area(research_area)

    meta = analyze_general_dataset(df)

    core_plan = build_sadi_core_plan(
        df,
        dataset_type=dataset_type,
        research_area=research_area,
    )

    # =========================
    # Datos base
    # =========================
    meta["dataset_type"] = dataset_type
    meta["research_area"] = research_area

    meta["suggested_plan"] = {
        "recommended_analysis": core_plan.get("recommended_analysis", []),
        "recommended_plots": core_plan.get("recommended_plots", []),
        "narrative_focus": core_plan.get("narrative_focus", ""),
        "warnings": core_plan.get("warnings", []),
    }
    meta["quick_recommendations"] = core_plan.get("quick_recommendations", [])
    meta["priority_order"] = core_plan.get("priority_order", [])
    meta["research_area_suggested"] = core_plan.get("research_area_suggested")

    # =========================
    # Insights automáticos
    # =========================
    insights = []

    n_rows = int(meta.get("n_rows", 0) or 0)
    n_cols = int(meta.get("n_cols", 0) or 0)
    n_num = int(meta.get("n_num", 0) or 0)
    n_cat = int(meta.get("n_cat", 0) or 0)
    n_dt = int(meta.get("n_dt", 0) or 0)
    missing_global_pct = float(meta.get("missing_global_pct", 0.0) or 0.0)

    if n_rows > 0 and n_cols > 0:
        insights.append(
            f"El dataset contiene {n_rows} filas y {n_cols} columnas, lo que permite un análisis exploratorio inicial consistente."
        )

    if n_num > 0:
        insights.append(
            f"Se detectaron {n_num} variables numéricas, adecuadas para análisis descriptivo, correlacional y eventualmente predictivo."
        )

    if n_cat > 0:
        insights.append(
            f"Se identificaron {n_cat} variables categóricas, útiles para segmentación, frecuencias y comparaciones entre grupos."
        )

    if n_dt > 0:
        insights.append(
            f"Se encontraron {n_dt} variables de fecha o tiempo, lo que abre la posibilidad de análisis temporal."
        )

    if missing_global_pct == 0:
        insights.append(
            "No se detectaron valores faltantes a nivel global, lo que favorece análisis más directos y limpios."
        )
    elif missing_global_pct < 5:
        insights.append(
            f"El porcentaje global de valores faltantes es bajo ({missing_global_pct:.2f}%), por lo que su tratamiento sería manejable."
        )
    elif missing_global_pct < 20:
        insights.append(
            f"El dataset presenta valores faltantes moderados ({missing_global_pct:.2f}%), por lo que conviene revisar imputación o limpieza antes de análisis avanzados."
        )
    else:
        insights.append(
            f"El porcentaje de valores faltantes es elevado ({missing_global_pct:.2f}%), lo que puede afectar la robustez de los resultados si no se trata previamente."
        )

    # NUEVOS insights de correlaciones / variabilidad / outliers
    if meta.get("high_corr_pairs"):
        pair = meta["high_corr_pairs"][0]
        insights.append(
            f"Se detectó una relación fuerte entre '{pair['col1']}' y '{pair['col2']}' (r={pair['corr']})."
        )

    if meta.get("variable_importance"):
        top = meta["variable_importance"][0]
        insights.append(
            f"La variable '{top['column']}' destaca como la más relevante en términos de variabilidad y calidad de datos."
        )

    if meta.get("top_numeric_by_variability"):
        top_var = meta["top_numeric_by_variability"][0]
        insights.append(
            f"La variable '{top_var['column']}' presenta la mayor variabilidad observada en el dataset."
        )

    if meta.get("outlier_summary"):
        top_out = meta["outlier_summary"][0]
        insights.append(
            f"La variable '{top_out['column']}' concentra {top_out['n_outliers']} valores atípicos potenciales."
        )

    if meta.get("target_candidate") and meta.get("target_type") == "regression":
        insights.append(
            f"La variable '{meta['target_candidate']}' podría utilizarse como objetivo en un modelo de regresión."
        )

    if meta.get("target_candidate") and meta.get("target_type") == "classification":
        insights.append(
            f"La variable '{meta['target_candidate']}' podría utilizarse como objetivo en un modelo de clasificación."
        )
    if meta.get("model_suggestion"):
        insights.append(
            f"SADI sugiere aplicar: {meta['model_suggestion']}."
        )
    for note in meta.get("pattern_notes", []):
        insights.append(note)

    if dataset_type == "survey_likert":
        insights.append(
            "La estructura sugiere un dataset de encuesta tipo Likert, apropiado para análisis psicométrico, consistencia interna y estudio por dimensiones."
        )
    elif dataset_type == "survey_normal":
        insights.append(
            "La estructura sugiere una encuesta general, adecuada para frecuencias, cruces, comparaciones y análisis descriptivo aplicado."
        )
    elif dataset_type == "multivariate":
        insights.append(
            "La naturaleza multivariante del dataset lo hace apto para técnicas como PCA, clustering, regresión multivariable o reducción de dimensionalidad."
        )
    elif dataset_type == "qualitative":
        insights.append(
            "La configuración del dataset sugiere un enfoque cualitativo o mixto, por lo que convendría complementar con categorización temática e interpretación textual."
        )
    else:
        insights.append(
            "El dataset presenta una estructura general que permite comenzar con análisis exploratorio, visualizaciones y detección de patrones."
        )

    if research_area and research_area != "general":
        insights.append(
            f"El área de investigación declarada es '{research_area}', por lo que SADI puede orientar mejor las recomendaciones metodológicas."
        )

    meta["insights"] = insights[:10]

    # =========================
    # Conclusión narrativa
    # =========================
    narrative_parts = []

    if n_rows and n_cols:
        narrative_parts.append(
            f"El dataset analizado presenta {n_rows} registros y {n_cols} variables."
        )

    if n_num or n_cat or n_dt:
        narrative_parts.append(
            f"En su estructura se identifican {n_num} variables numéricas, {n_cat} categóricas y {n_dt} temporales."
        )

    if missing_global_pct == 0:
        narrative_parts.append(
            "No se observaron valores faltantes globales, lo que constituye una base favorable para el análisis."
        )
    else:
        narrative_parts.append(
            f"Además, el porcentaje global de datos faltantes alcanza {missing_global_pct:.2f}%, aspecto que debe considerarse antes de aplicar técnicas más complejas."
        )

    if core_plan.get("narrative_focus"):
        narrative_parts.append(
            f"Desde la perspectiva analítica, el enfoque sugerido por SADI es: {core_plan.get('narrative_focus')}"
        )

    if meta.get("high_corr_pairs"):
        narrative_parts.append(
            f"Se identificaron {len(meta['high_corr_pairs'])} relaciones fuertes entre variables numéricas."
        )

    if meta.get("outlier_summary"):
        narrative_parts.append(
            f"Además, se detectaron posibles outliers en {len(meta['outlier_summary'])} variable(s)."
        )

    if meta.get("top_numeric_by_variability"):
        narrative_parts.append(
            "Las variables más dispersas podrían explicar buena parte de la variabilidad observada."
        )
    if meta.get("target_candidate") and meta.get("target_type"):
        narrative_parts.append(
            f"SADI detectó como posible variable objetivo '{meta['target_candidate']}', dentro de un contexto de {meta['target_type']}."
        )
    meta["insights_text"] = " ".join(narrative_parts).strip()

    return meta
def build_next_step_recommendation(
    *,
    analysis_meta: dict | None = None,
    dataset_kind: str = "dataset",
    model_plots: list[str] | None = None,
) -> dict:
    analysis_meta = analysis_meta or {}
    model_plots = model_plots or []

    recommendation = {
        "title": "Continuar con análisis exploratorio",
        "reason": "SADI recomienda profundizar primero en la comprensión estructural del dataset antes de avanzar a técnicas más complejas.",
        "action": "Revisar correlaciones, outliers y variables clave.",
    }

    n_num = int(analysis_meta.get("n_num", 0) or 0)
    n_cat = int(analysis_meta.get("n_cat", 0) or 0)
    high_corr_pairs = analysis_meta.get("high_corr_pairs", []) or []
    outlier_summary = analysis_meta.get("outlier_summary", []) or []
    target_candidate = analysis_meta.get("target_candidate")
    target_type = analysis_meta.get("target_type")
    model_suggestion = analysis_meta.get("model_suggestion")

    has_rf = any("rf_" in str(p).lower() for p in model_plots)
    has_reg = any("regression_" in str(p).lower() for p in model_plots)
    has_predictive_models = has_rf or has_reg

    if dataset_kind in ("survey_likert_5", "survey_likert_7"):
        recommendation = {
            "title": "Profundizar en psicometría del instrumento",
            "reason": "Al tratarse de un dataset Likert, el siguiente paso natural es revisar consistencia interna, dimensiones y estructura factorial.",
            "action": "Revisar alfa de Cronbach, análisis por dimensión y EFA.",
        }
        return recommendation

    if dataset_kind == "multivariate":
        recommendation = {
            "title": "Comparar enfoques multivariados",
            "reason": "El dataset ya está en modo multivariante, por lo que conviene comparar reducción de dimensionalidad, clustering y modelos predictivos.",
            "action": "Revisar PCA, clustering, EFA y comparación entre modelos.",
        }
        return recommendation

    # dataset normal
    if target_candidate and target_type and not has_predictive_models:
        recommendation = {
            "title": "Ejecutar el modelo sugerido por SADI",
            "reason": f"SADI detectó la variable objetivo '{target_candidate}' en un contexto de {target_type}, por lo que el siguiente paso natural es probar el modelo recomendado.",
            "action": model_suggestion or "Ejecutar un modelo predictivo acorde al tipo de target.",
        }
        return recommendation

    if has_predictive_models and has_rf and has_reg:
        recommendation = {
            "title": "Comparar y validar modelos predictivos",
            "reason": "Ya existen resultados de enfoques lineales y no lineales, por lo que el siguiente paso es comparar desempeño, interpretación y robustez.",
            "action": "Contrastar R², error, importancia de variables y comportamiento residual.",
        }
        return recommendation

    if len(high_corr_pairs) >= 3 and n_num >= 4:
        recommendation = {
            "title": "Explorar reducción de dimensionalidad",
            "reason": "La presencia de múltiples correlaciones fuertes sugiere redundancia entre variables y posible estructura latente.",
            "action": "Aplicar PCA o análisis factorial exploratorio.",
        }
        return recommendation

    if len(outlier_summary) >= 2:
        recommendation = {
            "title": "Revisar valores atípicos antes de modelar",
            "reason": "Se detectaron outliers en varias variables, lo que podría afectar estabilidad, ajuste e interpretación.",
            "action": "Inspeccionar outliers y evaluar limpieza, winsorización o modelos robustos.",
        }
        return recommendation

    if n_num >= 3 and n_cat >= 1:
        recommendation = {
            "title": "Explorar diferencias entre grupos",
            "reason": "El dataset combina variables numéricas y categóricas, lo que permite comparar perfiles o medias entre grupos.",
            "action": "Aplicar comparaciones entre grupos o análisis segmentado.",
        }
        return recommendation

    if n_num >= 3:
        recommendation = {
            "title": "Profundizar en relaciones numéricas",
            "reason": "El dataset cuenta con suficientes variables numéricas para avanzar hacia análisis relacionales más ricos.",
            "action": "Revisar correlaciones, scatter plots y posible modelado predictivo.",
        }
        return recommendation

    if n_cat >= 2:
        recommendation = {
            "title": "Profundizar en segmentación categórica",
            "reason": "Predominan variables categóricas, por lo que conviene explorar frecuencias, cruces y perfiles.",
            "action": "Analizar distribuciones y comparaciones entre categorías.",
        }
        return recommendation

    return recommendation
def build_model_comparison_summary(
    *,
    regression_result: dict | None = None,
    rf_result: dict | None = None,
) -> dict:
    def _to_float(v):
        try:
            return float(v)
        except Exception:
            return None

    lr_r2 = _to_float((regression_result or {}).get("r2"))
    lr_mae = _to_float((regression_result or {}).get("mae"))
    lr_rmse = _to_float((regression_result or {}).get("rmse"))

    rf_r2 = _to_float((rf_result or {}).get("r2"))
    rf_mae = _to_float((rf_result or {}).get("mae"))
    rf_rmse = _to_float((rf_result or {}).get("rmse"))

    winner = None
    summary = []

    if lr_r2 is not None and rf_r2 is not None:
        if abs(rf_r2 - lr_r2) < 1e-6:
            winner = "tie"
            summary.append("Ambos modelos muestran un rendimiento muy similar en términos de R².")
        elif rf_r2 > lr_r2:
            winner = "random_forest"
            summary.append("Random Forest supera a la regresión lineal en capacidad explicativa.")
        else:
            winner = "linear_regression"
            summary.append("La regresión lineal supera a Random Forest en capacidad explicativa.")

    if lr_rmse is not None and rf_rmse is not None:
        if rf_rmse < lr_rmse:
            summary.append("Random Forest presenta menor error de predicción (RMSE).")
        elif lr_rmse < rf_rmse:
            summary.append("La regresión lineal presenta menor error de predicción (RMSE).")

    if lr_mae is not None and rf_mae is not None:
        if rf_mae < lr_mae:
            summary.append("Random Forest también reduce el error absoluto medio.")
        elif lr_mae < rf_mae:
            summary.append("La regresión lineal también reduce el error absoluto medio.")

    if winner == "random_forest":
        recommendation = "SADI recomienda priorizar Random Forest para este dataset."
    elif winner == "linear_regression":
        recommendation = "SADI recomienda priorizar la regresión lineal para este dataset."
    elif winner == "tie":
        recommendation = "SADI considera que ambos modelos son comparables; conviene elegir según interpretabilidad o contexto."
    else:
        recommendation = "No hay suficiente información para recomendar un modelo con seguridad."

    return {
        "available": bool(regression_result or rf_result),
        "linear": regression_result or {},
        "random_forest": rf_result or {},
        "winner": winner,
        "summary": summary,
        "recommendation": recommendation,
    }
def build_advanced_classification_interpretation(
    *,
    rf_result: dict | None = None,
) -> list[str]:
    notes = []

    def _to_float(v):
        try:
            return float(v)
        except Exception:
            return None

    acc = _to_float((rf_result or {}).get("accuracy"))
    precision = _to_float((rf_result or {}).get("precision"))
    recall = _to_float((rf_result or {}).get("recall"))
    f1 = _to_float((rf_result or {}).get("f1_score"))

    if acc is not None:
        if acc >= 0.85:
            notes.append("La clasificación presenta un desempeño alto en términos de exactitud global.")
        elif acc >= 0.70:
            notes.append("La clasificación presenta un desempeño bueno para un análisis aplicado.")
        else:
            notes.append("La exactitud global del modelo es limitada y conviene revisar predictores o balance de clases.")

    if precision is not None and recall is not None:
        if abs(precision - recall) <= 0.05:
            notes.append("La precisión y la recuperación se encuentran relativamente balanceadas.")
        elif precision > recall:
            notes.append("El modelo parece más conservador al clasificar, priorizando precisión sobre recuperación.")
        else:
            notes.append("El modelo parece priorizar recuperación, aunque puede aumentar los falsos positivos.")

    if f1 is not None:
        if f1 >= 0.80:
            notes.append("El equilibrio general entre precisión y recall es sólido.")
        elif f1 >= 0.60:
            notes.append("El equilibrio entre precisión y recall es aceptable, aunque mejorable.")
        else:
            notes.append("El balance general entre precisión y recall todavía es débil.")

    if not notes:
        notes.append("No hay suficiente información para interpretar el modelo de clasificación.")

    seen = set()
    deduped = []
    for x in notes:
        if x not in seen:
            deduped.append(x)
            seen.add(x)

    return deduped