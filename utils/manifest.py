from __future__ import annotations

import json
import os

from config import MANIFESTS_DIR
from blueprints.multivariate.services import run_rf_regression_analysis, run_rf_classification_analysis
from blueprints.dataset.analysis import auto_select_best_regression_model
from flask import current_app
def _manifest_path(dataset_id: int) -> str:
    return os.path.join(MANIFESTS_DIR, f"ds{dataset_id}.json")


def read_manifest_data(dataset_id: int) -> dict:
    try:
        path = _manifest_path(dataset_id)
        if not os.path.exists(path):
            return {"plots": [], "meta": {}}

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}

        if not isinstance(data, dict):
            return {"plots": [], "meta": {}}

        plots = data.get("plots") or []
        meta = data.get("meta") or {}

        if not isinstance(plots, list):
            plots = []
        plots = [p for p in plots if isinstance(p, str)]

        if not isinstance(meta, dict):
            meta = {}

        data["plots"] = plots
        data["meta"] = meta
        return data

    except Exception:
        return {"plots": [], "meta": {}}


def write_manifest(dataset_id: int, data: dict) -> None:
    path = _manifest_path(dataset_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if not isinstance(data, dict):
        data = {}

    plots = data.get("plots") or []
    meta = data.get("meta") or {}

    if not isinstance(plots, list):
        plots = []
    plots = [p for p in plots if isinstance(p, str)]

    if not isinstance(meta, dict):
        meta = {}

    data["plots"] = plots
    data["meta"] = meta

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def auto_run_sadi_model(
    *,
    df,
    dataset_id,
    plots_dir,
    analysis_meta,
    manifest_data,
):
    """
    Ejecuta automáticamente el modelo sugerido por SADI
    si aún no existe en el manifest.
    """
    if not isinstance(analysis_meta, dict):
        return manifest_data

    target = analysis_meta.get("target_candidate")
    target_type = analysis_meta.get("target_type")

    if not target or not target_type:
        current_app.logger.warning(
            f"[auto_run_sadi_model] ds{dataset_id}: sin target o target_type."
        )
        return manifest_data

    model_results = manifest_data.get("model_results", {})
    if not isinstance(model_results, dict):
        model_results = {}

    try:
        # =========================
        # REGRESIÓN
        # =========================
        if target_type == "regression":
            current_app.logger.warning(
                f"[auto_run_sadi_model] ds{dataset_id}: ejecutando selección inteligente de regresión con target='{target}'"
            )

            manifest_data = auto_select_best_regression_model(
                df=df,
                dataset_id=dataset_id,
                plots_dir=plots_dir,
                target_col=target,
                manifest_data=manifest_data,
            )

            model_results = manifest_data.get("model_results", {})
            if not isinstance(model_results, dict):
                model_results = {}

            current_app.logger.warning(
                f"[auto_run_sadi_model] ds{dataset_id}: model_results después de regresión = {list(model_results.keys())}"
            )

        # =========================
        # CLASIFICACIÓN
        # =========================
        elif target_type == "classification":
            current_app.logger.warning(
                f"[auto_run_sadi_model] ds{dataset_id}: ejecutando clasificación automática con target='{target}'"
            )

            if not model_results.get("rf_result"):
                rf_result = run_rf_classification_analysis(
                    df=df,
                    dataset_id=dataset_id,
                    plots_dir=plots_dir,
                    target_col=target,
                )
                model_results["rf_result"] = rf_result
                manifest_data["model_results"] = model_results

            current_app.logger.warning(
                f"[auto_run_sadi_model] ds{dataset_id}: model_results después de clasificación = {list(model_results.keys())}"
            )

    except Exception as e:
        current_app.logger.exception(f"[auto_run_sadi_model] ds{dataset_id}: {e}")

    return manifest_data
