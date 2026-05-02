import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import PLOTS_DIR
from utils.plot_manager import normalize_plot_catalog, split_plots

# =========================================================
# READ DATAFRAME
# =========================================================

def read_dataframe(path, delimiter=None):

    if path.lower().endswith(".csv") or path.lower().endswith(".txt"):
        return pd.read_csv(path, sep=delimiter or ",", encoding="utf-8")

    if path.lower().endswith(".xlsx") or path.lower().endswith(".xls"):
        return pd.read_excel(path)

    raise ValueError("Formato de archivo no soportado")


# =========================================================
# CLEAN LIKERT VALUE
# =========================================================

def coerce_likert_cell_to_int(v):

    if pd.isna(v):
        return None

    try:
        return int(v)
    except Exception:
        pass

    v = str(v).strip()

    mapping = {
        "muy en desacuerdo": 1,
        "en desacuerdo": 2,
        "neutral": 3,
        "de acuerdo": 4,
        "muy de acuerdo": 5,
    }

    v_lower = v.lower()

    if v_lower in mapping:
        return mapping[v_lower]

    try:
        return int(float(v))
    except Exception:
        return None


# =========================================================
# BUILD LIKERT ITEMS
# =========================================================

def build_likert_items_df(df):

    likert_df = df.copy()

    for c in likert_df.columns:
        likert_df[c] = likert_df[c].apply(coerce_likert_cell_to_int)

    likert_df = likert_df.dropna(axis=1, how="all")

    return likert_df


# =========================================================
# CRONBACH ALPHA
# =========================================================

def cronbach_alpha(df):

    df = df.dropna()

    k = df.shape[1]

    if k < 2:
        return None

    variances = df.var(axis=0, ddof=1)
    total_var = df.sum(axis=1).var(ddof=1)

    if total_var == 0:
        return None

    alpha = (k / (k - 1)) * (1 - variances.sum() / total_var)

    return float(alpha)


# =========================================================
# SUMMARY
# =========================================================

def compute_likert_summary(df):

    summary = {}

    item_means = df.mean()

    summary["items"] = []

    for col in df.columns:

        summary["items"].append({
            "item": col,
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
        })

    summary["mean_global"] = float(df.mean().mean())

    alpha = cronbach_alpha(df)

    summary["cronbach_alpha"] = alpha

    return summary


# =========================================================
# PLOT SUMMARY
# =========================================================

def plot_likert_summary_scores(df, dataset_id):

    means = df.mean()

    plt.figure(figsize=(10,6))

    means.sort_values().plot(kind="barh")

    plt.title("Promedio por ítem (Likert)")
    plt.xlabel("Promedio")
    plt.ylabel("Ítems")

    os.makedirs(PLOTS_DIR, exist_ok=True)

    filename = f"ds{dataset_id}_likert_summary.png"

    path = os.path.join(PLOTS_DIR, filename)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    return os.path.join("plots", filename)
def build_likert_dimensions(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Agrupa ítems Likert por dimensión usando el prefijo antes de '_' .
    Ejemplos:
      D1_P1, D1_P2 -> dimensión D1
      Satisfaccion_1, Satisfaccion_2 -> dimensión Satisfaccion
    Si no encuentra '_', usa una dimensión general.
    """
    dimensions: dict[str, list[str]] = {}

    for col in df.columns:
        name = str(col).strip()
        if "_" in name:
            dim = name.split("_")[0].strip()
        else:
            dim = "General"

        dimensions.setdefault(dim, []).append(col)

    # dejar solo dimensiones con al menos 1 ítem
    dimensions = {k: v for k, v in dimensions.items() if v}
    return dimensions


def compute_dimension_summary(df: pd.DataFrame) -> list[dict]:
    """
    Calcula promedio, desviación y número de ítems por dimensión.
    """
    dims = build_likert_dimensions(df)
    rows: list[dict] = []

    for dim, cols in dims.items():
        sub = df[cols].copy()
        item_means = sub.mean(axis=0, numeric_only=True)
        dim_score = sub.mean(axis=1, numeric_only=True)

        rows.append({
            "dimension": dim,
            "n_items": len(cols),
            "items": cols,
            "mean": float(dim_score.mean()) if not dim_score.empty else None,
            "std": float(dim_score.std()) if not dim_score.empty else None,
            "item_means": {c: float(item_means[c]) for c in item_means.index},
        })

    rows.sort(key=lambda x: (x["mean"] is None, -(x["mean"] or 0)))
    return rows


def plot_dimension_summary_scores(df: pd.DataFrame, dataset_id: int) -> str | None:
    import numpy as np
    """
    Gráfico de barras horizontales con promedio por dimensión.
    """
    try:
        summary = compute_dimension_summary(df)
        if not summary:
            return None

        labels = [row["dimension"] for row in summary]
        values = [row["mean"] for row in summary]

        fig, ax = plt.subplots(figsize=(9, max(4, len(labels) * 0.55)))
        y = np.arange(len(labels))

        ax.barh(y, values)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("Promedio")
        ax.set_title("Promedio por dimensión")

        for i, v in enumerate(values):
            if v is not None:
                ax.text(v + 0.03, i, f"{v:.2f}", va="center", fontsize=9)

        out_name = f"ds{dataset_id}_dimension_summary_scores.png"
        out_path = os.path.join(PLOTS_DIR, out_name)
        fig.tight_layout()
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        return out_name
    except Exception as e:
        current_app.logger.warning(f"[plot_dimension_summary_scores] ds{dataset_id}: {e}")
        return None


def plot_dimension_radar(df: pd.DataFrame, dataset_id: int, scale: int = 5) -> str | None:
    import numpy as np
    """
    Gráfico radar con promedio por dimensión.
    """
    try:
        summary = compute_dimension_summary(df)
        if not summary or len(summary) < 2:
            return None

        labels = [row["dimension"] for row in summary]
        values = [float(row["mean"] or 0) for row in summary]

        # cerrar el radar
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        labels += labels[:1]

        fig = plt.figure(figsize=(7, 7))
        ax = plt.subplot(111, polar=True)

        ax.plot(angles, values, linewidth=2)
        ax.fill(angles, values, alpha=0.20)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels[:-1])
        ax.set_ylim(0, scale)
        ax.set_yticks(range(1, scale + 1))
        ax.set_title("Radar de dimensiones", pad=20)

        out_name = f"ds{dataset_id}_dimension_radar.png"
        out_path = os.path.join(PLOTS_DIR, out_name)
        fig.tight_layout()
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        return out_name
    except Exception as e:
        current_app.logger.warning(f"[plot_dimension_radar] ds{dataset_id}: {e}")
        return None
    