import os

def normalize_plot_catalog(dataset_id: int, manifest_data: dict, summary: dict | None = None) -> list:
    """
    Fuente única de verdad de gráficos para SADI.

    Devuelve una lista única y ordenada de rutas relativas tipo:
        plots/ds12_algo.png

    Reúne gráficos desde:
    - manifest_data["generated"]
    - manifest_data["plots"]
    - summary["plots"]
    - keys conocidas en summary/meta
    - escaneo físico de la carpeta plots para ds{dataset_id}_*.png
    """
    import os
    import re

    summary = summary or {}
    if not isinstance(summary, dict):
        summary = {}

    manifest_data = manifest_data or {}
    if not isinstance(manifest_data, dict):
        manifest_data = {}

    plots = []
    seen = set()

    # Ajusta esta ruta si en tu proyecto PLOTS_DIR ya existe globalmente
    plots_dir = PLOTS_DIR if "PLOTS_DIR" in globals() else os.path.join("static", "plots")

    def add_plot(p):
        if not isinstance(p, str):
            return

        rel = p.replace("\\", "/").strip()
        if not rel:
            return

        fname = rel.split("/")[-1]

        # solo PNG del dataset actual
        if not fname.lower().endswith(".png"):
            return
        if not fname.startswith(f"ds{dataset_id}_"):
            return

        full_path = os.path.join(plots_dir, fname)
        if not os.path.exists(full_path):
            return

        rel_final = f"plots/{fname}"
        if fname not in seen:
            plots.append(rel_final)
            seen.add(fname)

    # =========================
    # 1) manifest
    # =========================
    for p in manifest_data.get("generated", []) or []:
        add_plot(p)

    for p in manifest_data.get("plots", []) or []:
        add_plot(p)

    # =========================
    # 2) summary directo
    # =========================
    for p in summary.get("plots", []) or []:
        add_plot(p)

    # =========================
    # 3) keys conocidas de summary/meta
    # =========================
    known_keys = [
        # Likert
        "cronbach_plot",
        "summary_plot",
        "divergent_plot",
        "dimension_summary_plot",
        "dimension_radar_plot",

        # Psicometría avanzada
        "scree_plot",
        "factor_loadings_plot",
        "corr_heatmap_plot",
        "factor_model_plot",

        # Dataset / general
        "correlation_heatmap_plot",
        "missing_plot",
        "box_plot",
        "scatter_plot",
        "hist_plot",
        "bar_plot",

        # Multivariante / grupos / PCA
        "centroids_plot",
        "group_centroids_plot",
        "centroid_distance_plot",
        "pca_groups_plot",
        "pca_groups_ellipses_plot",

        # Predictivos
        "regression_coefficients_plot",
        "regression_pred_vs_real_plot",
        "regression_residuals_plot",
        "rf_feature_importance_plot",
        "rf_pred_vs_real_plot",
        "roc_curve_plot",
        "confusion_matrix_plot",
    ]

    for key in known_keys:
        add_plot(summary.get(key))
        add_plot(manifest_data.get(key))

    # =========================
    # 4) keys dentro de analysis_meta si existe
    # =========================
    analysis_meta = summary.get("analysis_meta", {}) or {}
    if isinstance(analysis_meta, dict):
        for key in known_keys:
            add_plot(analysis_meta.get(key))

        for p in analysis_meta.get("plots", []) or []:
            add_plot(p)

    # =========================
    # 5) escaneo físico real de la carpeta
    #    -> este es el que te va a recuperar heatmap,
    #       centroides, distancia centroides, PCA elipses, etc.
    # =========================
    try:
        ds_re = re.compile(rf"^ds{dataset_id}_.+\.png$", re.IGNORECASE)
        all_png = [f for f in os.listdir(plots_dir) if f.lower().endswith(".png")]
        found = [f for f in all_png if ds_re.match(f)]

        for fname in found:
            add_plot(fname)
    except Exception as e:
        if "current_app" in globals():
            current_app.logger.warning(f"[normalize_plot_catalog] ds{dataset_id}: {e}")

    # =========================
    # 6) orden científico consistente
    # =========================
    def order_key(rel_path: str):
        fname = rel_path.split("/")[-1].lower()

        # Exploratorios / calidad / psicometría
        if "missing" in fname:
            return (0, fname)
        if "cronbach" in fname:
            return (1, fname)
        if "likert_summary" in fname or "summary_scores" in fname:
            return (2, fname)
        if "dimension_summary" in fname:
            return (3, fname)
        if "dimension_radar" in fname:
            return (4, fname)
        if "divergent" in fname:
            return (5, fname)
        if "corr_heatmap" in fname or ("heatmap" in fname and "corr" in fname):
            return (6, fname)
        if "scree" in fname:
            return (7, fname)
        if "factor_loadings" in fname:
            return (8, fname)
        if "factor_model" in fname:
            return (9, fname)
        if "centroid" in fname:
            return (10, fname)
        if "distance" in fname and "centroid" in fname:
            return (11, fname)
        if "pca" in fname:
            return (12, fname)
        if "box" in fname:
            return (13, fname)
        if "scatter" in fname:
            return (14, fname)
        if "hist" in fname:
            return (15, fname)
        if "bar" in fname:
            return (16, fname)

        # Predictivos
        if "regression_coefficients" in fname:
            return (30, fname)
        if "regression_pred_vs_real" in fname:
            return (31, fname)
        if "regression_residuals" in fname:
            return (32, fname)
        if "rf_feature_importance" in fname:
            return (33, fname)
        if "rf_pred_vs_real" in fname:
            return (34, fname)
        if "roc" in fname:
            return (35, fname)
        if "confusion" in fname:
            return (36, fname)

        return (99, fname)

    plots.sort(key=order_key)
    return plots


def split_plots(plot_list: list):
    exploratory = []
    model = []

    for p in plot_list or []:
        low = str(p).lower()

        if any(x in low for x in [
            "regression_coefficients",
            "regression_pred_vs_real",
            "regression_residuals",
            "rf_feature_importance",
            "rf_pred_vs_real",
            "roc",
            "confusion",
            "logistic",
            "classification",
        ]):
            model.append(p)
        else:
            exploratory.append(p)

    return exploratory, model