from __future__ import annotations

import os
import re
import math
from collections import Counter
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from blueprints.dataset.analysis import read_dataframe

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency, kruskal
from docx.shared import Inches

def read_dataframe(path: str, delimiter: str | None = None) -> pd.DataFrame:
    import io
    import os
    import csv
    import pandas as pd

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    ext = os.path.splitext(path)[1].lower()

    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    if ext == ".json":
        return pd.read_json(path)

    if ext == ".sav":
        try:
            import pyreadstat
        except ImportError:
            raise ValueError("Para leer archivos SAV necesitas instalar pyreadstat.")
        df, _ = pyreadstat.read_sav(path)
        return df

    if ext not in [".csv", ".txt"]:
        raise ValueError(f"Formato no soportado: {ext}")

    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

    def _try_read_from_text(text: str, explicit_sep: str | None = None) -> pd.DataFrame | None:
        candidates = []

        if explicit_sep:
            candidates.append(explicit_sep)

        # sniff
        try:
            dialect = csv.Sniffer().sniff(text[:4096], delimiters=";,|\t,:")
            if dialect.delimiter not in candidates:
                candidates.append(dialect.delimiter)
        except Exception:
            pass

        # comunes
        for sep in [",", ";", "\t", "|", ":"]:
            if sep not in candidates:
                candidates.append(sep)

        best_df = None
        best_cols = 0

        for sep in candidates:
            try:
                df = pd.read_csv(io.StringIO(text), sep=sep, engine="python")
                if df.shape[1] > best_cols:
                    best_df = df
                    best_cols = df.shape[1]
                if df.shape[1] > 1:
                    return df
            except Exception:
                continue

        try:
            df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
            if df.shape[1] > best_cols:
                best_df = df
        except Exception:
            pass

        return best_df

    last_err = None

    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, errors="replace") as f:
                raw = f.read()

            if not raw.strip():
                raise ValueError("El archivo está vacío.")

            sep = delimiter.strip() if delimiter else None
            if sep == r"\t":
                sep = "\t"

            # 1) lectura normal
            df = _try_read_from_text(raw, sep)
            if df is not None and df.shape[1] > 1:
                return df

            # 2) reparación: \n literales -> saltos reales
            repaired = raw
            changed = False

            if "\\n" in repaired:
                repaired = repaired.replace("\\n", "\n")
                changed = True

            if "\\t" in repaired:
                repaired = repaired.replace("\\t", "\t")
                changed = True

            if changed:
                df = _try_read_from_text(repaired, sep)
                if df is not None and df.shape[1] > 1:
                    return df

            # 3) devolver aunque sea una columna si no hay más remedio
            if df is not None:
                return df

        except Exception as e:
            last_err = e

    raise ValueError(f"No se pudo leer el CSV/TXT. Último error: {last_err}")

def repair_broken_csv_file(path: str, delimiter: str | None = None) -> dict:
    """
    Repara archivos CSV/TXT con problemas comunes:
    - saltos literales \\n
    - tabs literales \\t
    - contenido guardado como una sola línea

    Devuelve un dict con:
    {
        "ok": bool,
        "repaired": bool,
        "backup_path": str | None,
        "message": str
    }
    """
    import os
    import shutil

    if not os.path.exists(path):
        return {
            "ok": False,
            "repaired": False,
            "backup_path": None,
            "message": "El archivo no existe.",
        }

    ext = os.path.splitext(path)[1].lower()
    if ext not in [".csv", ".txt"]:
        return {
            "ok": False,
            "repaired": False,
            "backup_path": None,
            "message": "La reparación automática aplica solo a CSV/TXT.",
        }

    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    raw = None
    used_enc = None

    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, errors="replace") as f:
                raw = f.read()
            used_enc = enc
            break
        except Exception:
            continue

    if raw is None:
        return {
            "ok": False,
            "repaired": False,
            "backup_path": None,
            "message": "No se pudo leer el archivo para reparación.",
        }

    if not raw.strip():
        return {
            "ok": False,
            "repaired": False,
            "backup_path": None,
            "message": "El archivo está vacío.",
        }

    repaired = raw
    changed = False

    # reparar secuencias literales
    if "\\n" in repaired:
        repaired = repaired.replace("\\n", "\n")
        changed = True

    if "\\t" in repaired:
        repaired = repaired.replace("\\t", "\t")
        changed = True

    if "\\r" in repaired:
        repaired = repaired.replace("\\r", "\r")
        changed = True

    if not changed:
        return {
            "ok": True,
            "repaired": False,
            "backup_path": None,
            "message": "No se detectaron problemas reparables.",
        }

    # validar que el reparado ahora sí pueda leerse mejor
    try:
        df_before = read_dataframe(path, delimiter)
        before_cols = int(df_before.shape[1]) if df_before is not None else 0
    except Exception:
        before_cols = 0

    import io
    import pandas as pd
    import csv

    def _try_read_text(text: str, explicit_sep: str | None = None):
        candidates = []
        if explicit_sep:
            candidates.append(explicit_sep)

        try:
            dialect = csv.Sniffer().sniff(text[:4096], delimiters=";,|\t,:")
            if dialect.delimiter not in candidates:
                candidates.append(dialect.delimiter)
        except Exception:
            pass

        for sep in [",", ";", "\t", "|", ":"]:
            if sep not in candidates:
                candidates.append(sep)

        best_df = None
        best_cols = 0

        for sep in candidates:
            try:
                df = pd.read_csv(io.StringIO(text), sep=sep, engine="python")
                if df.shape[1] > best_cols:
                    best_df = df
                    best_cols = df.shape[1]
                if df.shape[1] > 1:
                    return df
            except Exception:
                continue

        try:
            df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
            if df.shape[1] > best_cols:
                best_df = df
        except Exception:
            pass

        return best_df

    sep = delimiter.strip() if delimiter else None
    if sep == r"\t":
        sep = "\t"

    try:
        df_after = _try_read_text(repaired, sep)
        after_cols = int(df_after.shape[1]) if df_after is not None else 0
    except Exception:
        after_cols = 0

    if after_cols < max(2, before_cols):
        return {
            "ok": False,
            "repaired": False,
            "backup_path": None,
            "message": "Se intentó reparar, pero el resultado no mejoró la estructura del archivo.",
        }

    backup_path = path + ".bak"
    try:
        if not os.path.exists(backup_path):
            shutil.copy2(path, backup_path)

        with open(path, "w", encoding=used_enc or "utf-8", newline="") as f:
            f.write(repaired)

        return {
            "ok": True,
            "repaired": True,
            "backup_path": backup_path,
            "message": "Archivo reparado correctamente.",
        }
    except Exception as e:
        return {
            "ok": False,
            "repaired": False,
            "backup_path": None,
            "message": f"No se pudo guardar el archivo reparado: {e}",
        }

# =========================================================
# HELPERS GENERALES
# =========================================================

def safe_text(value: Any, default: str = "—") -> str:
    if value is None:
        return default
    txt = str(value).strip()
    return txt if txt else default


def safe_add_picture(doc, image_path: str, width_inches: float = 6.0) -> bool:
    try:
        if image_path and os.path.exists(image_path):
            doc.add_picture(image_path, width=Inches(width_inches))
            return True
    except Exception:
        pass
    return False


def is_probably_id_column(name: str) -> bool:
    n = str(name).strip().lower()
    bad = {
        "id", "codigo", "cod", "nro", "numero", "index", "idx",
        "fila", "registro", "dni", "cedula", "ruc", "email"
    }
    return n in bad or n.endswith("_id") or n.startswith("id_")


def to_serializable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def is_numeric_series(s: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(s):
        return True

    try:
        converted = pd.to_numeric(s, errors="coerce")
        ratio = converted.notna().mean() if len(s) else 0
        return ratio >= 0.8
    except Exception:
        return False


def normalize_numeric_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s
    return pd.to_numeric(s, errors="coerce")


def normalize_category_value(v: Any) -> str:
    if pd.isna(v):
        return "Sin dato"
    txt = str(v).strip()
    return txt if txt else "Sin dato"


# =========================================================
# SEGMENTACIÓN
# =========================================================

def detect_best_segment_column(df: pd.DataFrame) -> str | None:
    """
    Intenta detectar una variable categórica útil para comparar grupos.
    """
    best_col = None
    best_score = -1.0

    n_rows = len(df)
    if n_rows == 0:
        return None

    for col in df.columns:
        if is_probably_id_column(col):
            continue

        s = df[col]
        if is_numeric_series(s):
            continue

        clean = s.dropna().astype(str).str.strip()
        if clean.empty:
            continue

        nunique = clean.nunique(dropna=True)
        valid_ratio = clean.shape[0] / n_rows

        # útil si tiene entre 2 y 12 grupos aprox
        if nunique < 2 or nunique > 12:
            continue

        # penalizar columnas casi únicas
        unique_ratio = nunique / max(1, clean.shape[0])
        if unique_ratio > 0.3:
            continue

        # equilibrio de grupos
        vc = clean.value_counts(normalize=True)
        balance = 1.0 - float(vc.max())  # mejor cuando no domina una sola categoría

        score = (valid_ratio * 0.45) + ((1 / nunique) * 0.10) + (balance * 0.45)

        if score > best_score:
            best_score = score
            best_col = col

    return best_col


# =========================================================
# RESUMEN DE COLUMNAS
# =========================================================

def summarize_column(series: pd.Series) -> dict:
    s = series.copy()
    name = str(series.name) if series.name is not None else "columna"

    n_total = int(len(s))
    n_missing = int(s.isna().sum())
    n_valid = int(n_total - n_missing)
    missing_pct = (n_missing / n_total * 100.0) if n_total else 0.0

    result = {
        "name": name,
        "n_total": n_total,
        "n_valid": n_valid,
        "n_missing": n_missing,
        "missing_pct": float(missing_pct),
        "dtype": "categorical",
        "interpretation": "",
    }

    if is_numeric_series(s):
        num = normalize_numeric_series(s).dropna()
        result["dtype"] = "numeric"
        result["numeric"] = {
            "mean": float(num.mean()) if not num.empty else None,
            "median": float(num.median()) if not num.empty else None,
            "min": float(num.min()) if not num.empty else None,
            "max": float(num.max()) if not num.empty else None,
            "std": float(num.std()) if not num.empty else None,
            "q1": float(num.quantile(0.25)) if not num.empty else None,
            "q3": float(num.quantile(0.75)) if not num.empty else None,
        }

        if not num.empty:
            mean_ = float(num.mean())
            std_ = float(num.std()) if not math.isnan(float(num.std())) else 0.0
            result["interpretation"] = (
                f"La variable {name} es numérica, con media {mean_:.2f} "
                f"y desviación estándar {std_:.2f}."
            )
        return result

    # categórica
    cat = s.map(normalize_category_value)
    vc = cat.value_counts(dropna=False)
    top_categories = []
    total_valid = max(1, int(vc.sum()))

    for idx, count in vc.head(10).items():
        pct = float(count / total_valid * 100.0)
        top_categories.append({
            "category": str(idx),
            "count": int(count),
            "pct": pct,
        })

    result["top_categories"] = top_categories

    if top_categories:
        top = top_categories[0]
        result["interpretation"] = (
            f"La categoría más frecuente en {name} fue '{top['category']}', "
            f"con {top['pct']:.1f}% de los casos válidos."
        )
    else:
        result["interpretation"] = f"La variable {name} no presenta categorías válidas suficientes."

    return result


# =========================================================
# CROSSTABS / ASOCIACIÓN
# =========================================================

def cramers_v_from_table(table: pd.DataFrame) -> float | None:
    try:
        chi2, _, _, _ = chi2_contingency(table)
        n = table.to_numpy().sum()
        if n == 0:
            return None
        r, k = table.shape
        denom = min(k - 1, r - 1)
        if denom <= 0:
            return None
        return float(np.sqrt((chi2 / n) / denom))
    except Exception:
        return None


def cramers_v_strength(v: float | None) -> str:
    if v is None:
        return ""
    if v < 0.10:
        return "muy débil"
    if v < 0.20:
        return "débil"
    if v < 0.40:
        return "moderada"
    if v < 0.60:
        return "fuerte"
    return "muy fuerte"


def cramers_v_class(v: float | None) -> str:
    if v is None:
        return ""
    if v >= 0.40:
        return "ct-strong"
    if v >= 0.20:
        return "ct-medium"
    return ""


def make_crosstab_tests(df: pd.DataFrame, segment_col: str) -> list[dict]:
    results = []

    if segment_col not in df.columns:
        return results

    seg = df[segment_col].map(normalize_category_value)

    for col in df.columns:
        if col == segment_col:
            continue

        s = df[col]
        if is_numeric_series(s):
            continue

        cat = s.map(normalize_category_value)

        table = pd.crosstab(seg, cat)
        if table.shape[0] < 2 or table.shape[1] < 2:
            continue

        try:
            chi2, p_value, _, _ = chi2_contingency(table)
            v = cramers_v_from_table(table)

            results.append({
                "variable": str(col),
                "chi2": float(chi2),
                "p_value": float(p_value),
                "cramers_v": float(v) if v is not None else None,
            })
        except Exception:
            continue

    return results


# =========================================================
# PRUEBAS NUMÉRICAS
# =========================================================

def numeric_group_tests(df: pd.DataFrame, segment_col: str) -> list[dict]:
    results = []

    if segment_col not in df.columns:
        return results

    seg = df[segment_col].map(normalize_category_value)

    for col in df.columns:
        if col == segment_col:
            continue

        s = df[col]
        if not is_numeric_series(s):
            continue

        num = normalize_numeric_series(s)
        temp = pd.DataFrame({"seg": seg, "num": num}).dropna()

        if temp.empty:
            continue

        groups = []
        for _, g in temp.groupby("seg"):
            vals = g["num"].dropna().values
            if len(vals) >= 2:
                groups.append(vals)

        if len(groups) < 2:
            continue

        try:
            stat, p_value = kruskal(*groups)
            results.append({
                "variable": str(col),
                "test": "Kruskal-Wallis",
                "statistic": float(stat),
                "p_value": float(p_value),
                "significant": bool(p_value < 0.05),
            })
        except Exception:
            continue

    return results


# =========================================================
# COMPARACIÓN ENTRE GRUPOS
# =========================================================

def build_group_comparisons(df: pd.DataFrame, segment_col: str) -> list[dict]:
    results = []

    if segment_col not in df.columns:
        return results

    seg = df[segment_col].map(normalize_category_value)

    for col in df.columns:
        if col == segment_col:
            continue

        s = df[col]
        if not is_numeric_series(s):
            continue

        num = normalize_numeric_series(s)
        temp = pd.DataFrame({"seg": seg, "num": num}).dropna()

        if temp.empty:
            continue

        group_rows = []
        arrays = []

        for group_name, g in temp.groupby("seg"):
            vals = g["num"].dropna()
            if vals.empty:
                continue

            arrays.append(vals.values)
            group_rows.append({
                "group": str(group_name),
                "n": int(vals.shape[0]),
                "mean": float(vals.mean()),
                "median": float(vals.median()),
                "std": float(vals.std()) if vals.shape[0] > 1 else 0.0,
            })

        if len(group_rows) < 2:
            continue

        p_value = None
        statistic = None
        try:
            statistic, p_value = kruskal(*arrays)
        except Exception:
            pass

        ordered = sorted(group_rows, key=lambda x: x["mean"], reverse=True)
        top_group = ordered[0]["group"]
        bottom_group = ordered[-1]["group"]

        interpretation = (
            f"La variable {col} muestra diferencias descriptivas entre grupos de {segment_col}. "
            f"El grupo con mayor media fue '{top_group}' y el de menor media fue '{bottom_group}'."
        )
        if isinstance(p_value, (int, float)):
            if p_value < 0.05:
                interpretation += " La diferencia fue estadísticamente significativa."
            else:
                interpretation += " No se observó significación estadística al nivel 0.05."

        results.append({
            "variable": str(col),
            "segment_column": str(segment_col),
            "groups": group_rows,
            "statistic": float(statistic) if isinstance(statistic, (int, float)) else None,
            "p_value": float(p_value) if isinstance(p_value, (int, float)) else None,
            "is_significant": bool(isinstance(p_value, (int, float)) and p_value < 0.05),
            "interpretation": interpretation,
        })

    return results


def build_group_comparison_summary(group_comparisons: list[dict]) -> str:
    if not group_comparisons:
        return ""

    sig = [g for g in group_comparisons if g.get("is_significant")]
    if sig:
        top = sig[:5]
        vars_txt = ", ".join(str(x.get("variable")) for x in top)
        return (
            "Se identificaron diferencias entre grupos en varias variables numéricas. "
            f"Las comparaciones más destacadas incluyen: {vars_txt}."
        )

    return (
        "Se realizaron comparaciones entre grupos en variables numéricas, "
        "pero no se detectaron diferencias estadísticamente significativas en los principales contrastes."
    )


def build_group_findings(group_comparisons: list[dict]) -> list[str]:
    findings = []

    ordered = sorted(
        group_comparisons,
        key=lambda x: (1 if x.get("is_significant") else 0, -(x.get("statistic") or 0)),
        reverse=True,
    )

    for item in ordered[:6]:
        var = item.get("variable", "variable")
        groups = item.get("groups") or []
        if not groups:
            continue

        groups_ordered = sorted(groups, key=lambda x: x.get("mean", 0), reverse=True)
        top_group = groups_ordered[0]
        findings.append(
            f"En {var}, el grupo '{top_group.get('group')}' presentó la media más alta "
            f"({top_group.get('mean'):.2f})."
        )

    return findings


# =========================================================
# VARIABLES IMPORTANTES / TEXTO AUTOMÁTICO
# =========================================================

def detect_key_variables(meta: dict) -> list[str]:
    important = []

    for x in meta.get("crosstabs", []) or []:
        if x.get("is_significant"):
            important.append(str(x.get("variable")))

    for x in meta.get("group_comparisons", []) or []:
        if x.get("is_significant"):
            important.append(str(x.get("variable")))

    # quitar duplicados preservando orden
    seen = set()
    clean = []
    for v in important:
        if v not in seen:
            clean.append(v)
            seen.add(v)

    return clean[:10]


def generate_results_text_survey_normal(meta: dict) -> str:
    n_total = meta.get("n_total", "—")
    segment_col = meta.get("segment_column")
    n_cols = len(meta.get("columns_summary", []) or [])
    important = meta.get("important_variables", []) or []

    parts = []
    parts.append(
        f"El análisis se realizó sobre {n_total} registros y {n_cols} variables."
    )

    if segment_col:
        parts.append(
            f"Se detectó como variable principal de segmentación a '{segment_col}', "
            "utilizada para explorar diferencias y asociaciones entre grupos."
        )
    else:
        parts.append(
            "No se detectó una variable de segmentación suficientemente robusta, "
            "por lo que el análisis se concentró en la descripción general del dataset."
        )

    if important:
        parts.append(
            "Las variables con mayor relevancia analítica fueron: "
            + ", ".join(important[:6]) + "."
        )

    crosstabs = meta.get("crosstabs", []) or []
    sig_ct = [x for x in crosstabs if x.get("is_significant")]
    if sig_ct:
        parts.append(
            "Se identificaron asociaciones estadísticamente significativas entre algunas variables categóricas."
        )

    group_comparisons = meta.get("group_comparisons", []) or []
    sig_gc = [x for x in group_comparisons if x.get("is_significant")]
    if sig_gc:
        parts.append(
            "También se observaron diferencias entre grupos en determinadas variables numéricas."
        )

    return "\n\n".join(parts)


def build_survey_insights(meta: dict) -> str:
    lines = []

    profile = meta.get("survey_profile") or {}
    n_total = profile.get("n_total", meta.get("n_total", "—"))
    n_var = profile.get("n_variables", "—")
    seg = profile.get("segment_column", meta.get("segment_column"))

    lines.append(f"Se analizaron {n_total} respuestas/registros y {n_var} variables.")

    if seg:
        lines.append(f"La variable de segmentación más útil fue '{seg}'.")

    important = meta.get("important_variables", []) or []
    if important:
        lines.append(
            "Las variables con mayor peso analítico fueron: " + ", ".join(important[:5]) + "."
        )

    crosstabs = meta.get("crosstabs", []) or []
    sig_ct = [x for x in crosstabs if x.get("is_significant")]
    if sig_ct:
        top = sorted(sig_ct, key=lambda x: x.get("cramers_v") or 0, reverse=True)[:3]
        assoc = ", ".join(
            f"{x.get('variable')} (V={x.get('cramers_v'):.3f})"
            for x in top
            if isinstance(x.get("cramers_v"), (int, float))
        )
        if assoc:
            lines.append(f"Las asociaciones categóricas más relevantes fueron: {assoc}.")

    group_comparisons = meta.get("group_comparisons", []) or []
    sig_gc = [x for x in group_comparisons if x.get("is_significant")]
    if sig_gc:
        vars_txt = ", ".join(str(x.get("variable")) for x in sig_gc[:4])
        lines.append(f"Se detectaron diferencias entre grupos en: {vars_txt}.")

    return "\n".join(lines)


def build_survey_key_findings(meta: dict) -> list[str]:
    findings = []

    if meta.get("segment_column"):
        findings.append(f"Se detectó '{meta.get('segment_column')}' como variable principal de segmentación.")

    important = meta.get("important_variables", []) or []
    for v in important[:4]:
        findings.append(f"La variable '{v}' mostró relevancia analítica destacada.")

    sig_ct = [x for x in (meta.get("crosstabs") or []) if x.get("is_significant")]
    if sig_ct:
        top = sorted(sig_ct, key=lambda x: x.get("cramers_v") or 0, reverse=True)[0]
        findings.append(
            f"La asociación categórica más fuerte se observó en '{top.get('variable')}'."
        )

    sig_gc = [x for x in (meta.get("group_comparisons") or []) if x.get("is_significant")]
    if sig_gc:
        findings.append(
            f"Se encontraron diferencias significativas entre grupos en {len(sig_gc)} variables numéricas."
        )

    return findings[:8]


# =========================================================
# GRÁFICOS
# =========================================================

def slugify_for_plot(text: str) -> str:
    txt = str(text)
    txt = re.sub(r"[^\w\s-]", "", txt, flags=re.UNICODE)
    txt = re.sub(r"[\s/]+", "_", txt)
    return txt[:80]


def plot_survey_normal_question(series: pd.Series, dataset_id: int, plots_dir: str) -> str | None:
    os.makedirs(plots_dir, exist_ok=True)
    col = str(series.name) if series.name is not None else "columna"
    safe_col = slugify_for_plot(col)

    try:
        if is_numeric_series(series):
            s = normalize_numeric_series(series).dropna()
            if s.empty:
                return None

            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            ax.hist(s, bins=min(12, max(5, int(math.sqrt(len(s))))), edgecolor="black")
            ax.set_title(f"Distribución: {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frecuencia")

            fname = f"ds{dataset_id}_hist_{safe_col}.png"
            fpath = os.path.join(plots_dir, fname)
            fig.tight_layout()
            fig.savefig(fpath, dpi=160)
            plt.close(fig)
            return fname

        s = series.map(normalize_category_value)
        vc = s.value_counts().head(10)
        if vc.empty:
            return None

        fig, ax = plt.subplots(figsize=(8, 4.8))
        vc.sort_values().plot(kind="barh", ax=ax)
        ax.set_title(f"Frecuencias: {col}")
        ax.set_xlabel("Frecuencia")
        ax.set_ylabel("Categoría")

        fname = f"ds{dataset_id}_bar_{safe_col}.png"
        fpath = os.path.join(plots_dir, fname)
        fig.tight_layout()
        fig.savefig(fpath, dpi=160)
        plt.close(fig)
        return fname

    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass
        return None


def generate_group_comparison_plots(
    *,
    df: pd.DataFrame,
    dataset_id: int,
    segment_col: str,
    group_comparisons: list[dict],
    plots_dir: str,
    max_plots: int = 8,
) -> list[str]:
    files = []
    os.makedirs(plots_dir, exist_ok=True)

    selected = group_comparisons[:max_plots]

    for comp in selected:
        variable = comp.get("variable")
        groups = comp.get("groups") or []
        if not variable or not groups:
            continue

        try:
            labels = [str(g.get("group")) for g in groups]
            means = [float(g.get("mean")) for g in groups]

            fig, ax = plt.subplots(figsize=(8, 4.8))
            ax.bar(labels, means)
            ax.set_title(f"Comparación de grupos: {variable}")
            ax.set_xlabel(segment_col)
            ax.set_ylabel("Media")

            plt.xticks(rotation=25, ha="right")

            fname = f"ds{dataset_id}_group_mean_{slugify_for_plot(variable)}.png"
            fpath = os.path.join(plots_dir, fname)
            fig.tight_layout()
            fig.savefig(fpath, dpi=160)
            plt.close(fig)

            files.append(fname)
        except Exception:
            try:
                plt.close("all")
            except Exception:
                pass

    return files


def filter_plots_for_dataset(dataset_id: int, plots) -> list[str]:
    if not isinstance(plots, list):
        return []

    prefix = f"ds{dataset_id}_"
    clean = []

    for p in plots:
        if not p:
            continue
        rel = str(p).replace("\\", "/").strip()
        fname = rel.split("/")[-1]
        if fname.startswith(prefix):
            clean.append(rel)

    return clean


# =========================================================
# METADATOS DE PLOTS
# =========================================================

def classify_plot_tag(plot_path: str) -> str:
    name = os.path.basename(str(plot_path)).lower()
    clean = re.sub(r"^ds\d+_", "", name)
    clean = re.sub(r"\.png$", "", clean)

    if clean.startswith("bar_") or clean.startswith("hist_"):
        return "Descriptivo"
    if clean.startswith("group_mean_"):
        return "Comparativo"
    return "Analítico"


def describe_plot(plot_path: str) -> str:
    name = os.path.basename(str(plot_path)).lower()
    clean = re.sub(r"^ds\d+_", "", name)
    clean = re.sub(r"\.png$", "", clean)

    if clean.startswith("bar_"):
        return "Resume las categorías más frecuentes de la variable analizada."
    if clean.startswith("hist_"):
        return "Muestra la distribución de la variable numérica analizada."
    if clean.startswith("group_mean_"):
        return "Compara promedios entre grupos detectados en la variable de segmentación."
    return "Gráfico generado automáticamente por SADI."


def prettify_plot_title(plot_path: str) -> str:
    name = os.path.basename(str(plot_path)).lower()
    clean = re.sub(r"^ds\d+_", "", name)
    clean = re.sub(r"\.png$", "", clean)

    if clean.startswith("bar_"):
        return f"Frecuencias: {clean[4:].replace('_', ' ').title()}"
    if clean.startswith("hist_"):
        return f"Distribución: {clean[5:].replace('_', ' ').title()}"
    if clean.startswith("group_mean_"):
        return f"Comparación por grupos: {clean[11:].replace('_', ' ').title()}"

    return clean.replace("_", " ").title()


def summarize_plot_tags(plots: list[str]) -> list[dict]:
    if not plots:
        return []

    counts = Counter()
    for p in plots:
        counts[classify_plot_tag(p)] += 1

    preferred = ["Descriptivo", "Comparativo", "Analítico"]

    result = []
    used = set()

    for tag in preferred:
        if tag in counts:
            result.append({"tag": tag, "count": int(counts[tag])})
            used.add(tag)

    for tag in sorted(counts.keys()):
        if tag not in used:
            result.append({"tag": tag, "count": int(counts[tag])})

    return result


# =========================================================
# CONTEXTO ACADÉMICO
# =========================================================

def build_academic_report_context(ds, meta: dict, plots: list[str]) -> dict:
    from datetime import datetime

    meta = meta or {}
    if not isinstance(meta, dict):
        meta = {}

    dataset_name = ds.title or ds.original_name or f"Dataset {getattr(ds, 'id', '')}"
    dataset_type = "Encuesta normal"
    n_total = meta.get("n_total", getattr(ds, "n_rows", "—"))

    survey_profile = meta.get("survey_profile") or {}
    if not isinstance(survey_profile, dict):
        survey_profile = {}

    columns_summary = meta.get("columns_summary") or []
    crosstabs = meta.get("crosstabs") or []
    group_comparisons = meta.get("group_comparisons") or []
    group_findings = meta.get("group_findings") or []
    key_findings = meta.get("survey_key_findings") or []
    important_variables = meta.get("important_variables") or []

    survey_insights = meta.get("survey_insights") or ""
    academic_results_text = meta.get("results_text") or ""
    crosstabs_insights = meta.get("crosstabs_insights") or ""
    group_comparison_summary = meta.get("group_comparison_summary") or ""

    # =========================
    # CAPA SADI AVANZADA
    # =========================
    sadi_insights = meta.get("sadi_insights") or ""
    sadi_recommendations = meta.get("sadi_recommendations") or meta.get("quick_recommendations") or []
    sadi_plan = meta.get("sadi_plan") or meta.get("suggested_plan") or {}
    sadi_priority = meta.get("sadi_priority") or meta.get("priority_order") or []
    next_step_recommendation = meta.get("next_step_recommendation") or {}
    warnings = meta.get("warnings") or []

    if not isinstance(sadi_recommendations, list):
        sadi_recommendations = [sadi_recommendations] if sadi_recommendations else []
    if not isinstance(sadi_plan, dict):
        sadi_plan = {}
    if not isinstance(sadi_priority, list):
        sadi_priority = [sadi_priority] if sadi_priority else []
    if not isinstance(next_step_recommendation, dict):
        next_step_recommendation = {}
    if not isinstance(warnings, list):
        warnings = [warnings] if warnings else []

    target_candidate = meta.get("target_candidate")
    target_type = meta.get("target_type")
    target_reason = meta.get("target_reason")
    model_suggestion = meta.get("model_suggestion")
    ranked_target_candidates = meta.get("ranked_target_candidates") or []
    variable_importance = meta.get("variable_importance") or []
    top_numeric_by_variability = meta.get("top_numeric_by_variability") or []

    # =========================
    # TEXTO CENTRAL
    # =========================
    intro_text = (
        f"El presente informe resume el análisis descriptivo y comparativo del dataset "
        f"'{dataset_name}', clasificado como encuesta normal. Se evaluaron frecuencias, "
        f"resúmenes descriptivos, asociaciones entre variables categóricas y, cuando fue posible, "
        f"diferencias entre grupos a partir de una variable de segmentación detectada automáticamente. "
        f"Además, SADI generó una lectura metodológica complementaria con recomendaciones, prioridades "
        f"analíticas y sugerencias de continuidad."
    )

    # prioriza lo avanzado
    conclusion_text = (
        sadi_insights
        or meta.get("insights_text")
        or survey_insights
        or "En conjunto, el análisis muestra un panorama descriptivo útil para comprender el comportamiento "
           "de las variables del estudio. Los resultados deben interpretarse junto con el contexto del instrumento, "
           "el diseño de recolección de datos y los objetivos específicos de la investigación."
    )

    descriptive_overview = (
        f"Se resumieron {len(columns_summary)} variables correspondientes al dataset analizado. "
        f"El sistema detectó {survey_profile.get('n_categorical', '—')} variables categóricas y "
        f"{survey_profile.get('n_numeric', '—')} numéricas."
    )

    return {
        "title": "Informe académico — Encuesta normal",
        "dataset_name": dataset_name,
        "file_name": getattr(ds, "original_name", "—"),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "dataset_type": dataset_type,
        "n_total": n_total,
        "segment_column": meta.get("segment_column"),
        "intro_text": intro_text,
        "conclusion_text": conclusion_text,

        # base survey
        "academic_results_text": academic_results_text,
        "columns_summary": columns_summary,
        "crosstabs": crosstabs,
        "crosstabs_insights": crosstabs_insights,
        "group_comparisons": group_comparisons,
        "group_comparison_summary": group_comparison_summary,
        "group_findings": group_findings,
        "key_findings": key_findings,
        "important_variables": important_variables,
        "survey_profile": survey_profile,
        "survey_key_findings": key_findings,
        "survey_insights": survey_insights,
        "descriptive_overview": descriptive_overview,

        # capa SADI
        "sadi_insights": sadi_insights,
        "sadi_recommendations": sadi_recommendations,
        "sadi_plan": sadi_plan,
        "sadi_priority": sadi_priority,
        "next_step_recommendation": next_step_recommendation,
        "warnings": warnings,

        # tablas extra
        "top_numeric_by_variability": top_numeric_by_variability,
        "variable_importance": variable_importance,

        # bloque predictivo sugerido
        "target_candidate": target_candidate,
        "target_type": target_type,
        "target_reason": target_reason,
        "model_suggestion": model_suggestion,
        "ranked_target_candidates": ranked_target_candidates,

        # soporte adicional
        "plots": plots or [],
        "insights_text": meta.get("insights_text") or "",
        "quick_recommendations": meta.get("quick_recommendations") or [],
        "suggested_plan": meta.get("suggested_plan") or {},
        "priority_order": meta.get("priority_order") or [],
    }


# =========================================================
# PDF ENCUESTA NORMAL
# =========================================================

def generate_survey_normal_report_pdf(
    *,
    dataset_id: int,
    dataset_title: str,
    manifest_data: dict,
    plots_dir: str,
    output_path: str,
) -> None:
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Image as RLImage,
        Table,
        TableStyle,
        PageBreak,
    )
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors

    if not isinstance(manifest_data, dict):
        manifest_data = {}

    meta = manifest_data.get("meta") or {}
    plots = manifest_data.get("generated") or manifest_data.get("plots") or []

    if not isinstance(meta, dict):
        meta = {}
    if not isinstance(plots, list):
        plots = []

    plots = normalize_plot_catalog(dataset_id, manifest_data, summary)
    exploratory_plots, model_plots = split_plots(plots)

    styles = getSampleStyleSheet()
    normal = styles["BodyText"]
    title_style = styles["Title"]
    h1 = styles["Heading1"]
    h2 = styles["Heading2"]

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    elements = []

    elements.append(Paragraph("Informe de encuesta normal", title_style))
    elements.append(Spacer(1, 0.4 * cm))
    elements.append(Paragraph(f"Dataset: {dataset_title}", normal))
    elements.append(Paragraph(f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M')}", normal))
    elements.append(Spacer(1, 0.4 * cm))

    # Resumen
    profile = meta.get("survey_profile") or {}
    table_data = [
        ["Indicador", "Valor"],
        ["Total de participantes", str(profile.get("n_total", meta.get("n_total", "—")))],
        ["Variables analizadas", str(profile.get("n_variables", "—"))],
        ["Variables categóricas", str(profile.get("n_categorical", "—"))],
        ["Variables numéricas", str(profile.get("n_numeric", "—"))],
        ["Segmento principal", str(profile.get("segment_column", meta.get("segment_column", "No detectado")))],
    ]

    t = Table(table_data, colWidths=[7 * cm, 7 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.5 * cm))

    # Insights
    if meta.get("survey_insights"):
        elements.append(Paragraph("Resumen ejecutivo", h1))
        for line in str(meta.get("survey_insights")).split("\n"):
            line = line.strip()
            if line:
                elements.append(Paragraph(line, normal))
        elements.append(Spacer(1, 0.4 * cm))

    # Hallazgos clave
    key_findings = meta.get("survey_key_findings") or []
    if key_findings:
        elements.append(Paragraph("Hallazgos clave", h1))
        for item in key_findings[:8]:
            elements.append(Paragraph(f"• {item}", normal))
        elements.append(Spacer(1, 0.4 * cm))

    # Crosstabs
    crosstabs = meta.get("crosstabs") or []
    if crosstabs:
        elements.append(Paragraph("Asociaciones entre variables categóricas", h1))

        table_data = [["Variable", "Chi²", "p-value", "Cramér’s V", "Interpretación"]]
        for r in crosstabs[:25]:
            table_data.append([
                str(r.get("variable", "")),
                str(r.get("chi2", "")),
                str(r.get("p_value", "")),
                str(r.get("cramers_v", "")),
                str(r.get("strength", "")),
            ])

        t2 = Table(table_data, repeatRows=1)
        t2.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
        ]))
        elements.append(t2)
        elements.append(Spacer(1, 0.5 * cm))

        if meta.get("crosstabs_insights"):
            elements.append(Paragraph("Interpretación automática", h2))
            for line in str(meta.get("crosstabs_insights")).split("\n"):
                line = line.strip()
                if line:
                    elements.append(Paragraph(line, normal))
            elements.append(Spacer(1, 0.4 * cm))

    # Resultados automáticos
    if meta.get("results_text"):
        elements.append(Paragraph("Resultados automáticos", h1))
        for block in str(meta.get("results_text")).split("\n\n"):
            block = block.strip()
            if block:
                elements.append(Paragraph(block, normal))
        elements.append(Spacer(1, 0.4 * cm))

    # Gráficos
    if plots:
        elements.append(PageBreak())
        elements.append(Paragraph("Gráficos del análisis", h1))

        for rel in plots:
            rel = str(rel).replace("\\", "/")
            fname = rel.split("/")[-1]
            img_path = os.path.join(plots_dir, fname)
            if not os.path.exists(img_path):
                continue

            elements.append(Paragraph(prettify_plot_title(rel), h2))
            elements.append(Paragraph(describe_plot(rel), normal))
            try:
                img = RLImage(img_path)
                img._restrictSize(16 * cm, 10 * cm)
                elements.append(img)
                elements.append(Spacer(1, 0.4 * cm))
            except Exception:
                continue

    doc.build(elements)