from __future__ import annotations

import os
import pandas as pd
from flask import (
    Blueprint,
    render_template,
    redirect,
    url_for,
    flash,
    current_app,
    request,
    session,
    send_file,
)
from flask_login import login_required, current_user

from extensions import SessionLocal
from models import Dataset
from config import UPLOAD_DIR, PLOTS_DIR
from blueprints.survey.analysis import read_dataframe

from blueprints.dataset.analysis import (
       analyze_dataset_with_recommendations
    )

from utils.plot_meta import (
    classify_plot_tag,
    describe_plot,
    prettify_plot_title,
    summarize_plot_tags,
    build_general_dataset_figure_catalog,
)
from utils.manifest import read_manifest_data, write_manifest
from utils.plot_manager import normalize_plot_catalog, split_plots
from .services import (
    build_multivariate_profile,
    generate_correlation_heatmap,
    run_pca_analysis,
    run_kmeans_analysis,
    run_efa_analysis,
    run_regression_analysis,
    run_rf_regression_analysis,
    run_outlier_analysis,
    run_full_sadi_analysis,
    generate_multivariate_report_docx,
    generate_multivariate_article_docx,
    generate_multivariate_interpretation,
    run_anova_analysis,
    run_manova_analysis,
    run_permanova_analysis,
    run_logistic_regression_analysis,
    generate_group_visualizations,
    detect_prediction_context,
    compare_regression_models,
    build_unified_multivariate_figure_catalog,
    get_multivariate_figure_catalog,
    get_or_build_multivariate_analysis_cache,
    generate_sadi_conclusion,
    generate_sadi_abstract_and_keywords,
    generate_sadi_limitations_and_future_work,
    generate_insights_ranking,
    run_rf_classification_analysis
)

multivariate_bp = Blueprint(
    "multivariate",
    __name__,
    url_prefix="/multivariate",
    template_folder="templates",
)


@multivariate_bp.get("/<int:dataset_id>")
@login_required
def multivariate_detail(dataset_id: int):
    db = SessionLocal()
    try:
        ds = db.get(Dataset, dataset_id)
        if not ds or ds.user_id != current_user.id:
            flash("Dataset no encontrado.", "warning")
            return redirect(url_for("dataset.dashboard"))

        research_type = getattr(ds, "research_type", "general")

        path = os.path.join(UPLOAD_DIR, ds.filename)
        df = read_dataframe(path, ds.delimiter)

        # =========================
        # Detectar columnas útiles
        # =========================
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        categorical_cols = [c for c in df.columns if c not in numeric_cols]

        regression_target_col = numeric_cols[-1] if len(numeric_cols) >= 3 else None
        rf_target_col = numeric_cols[-1] if len(numeric_cols) >= 3 else None

        logistic_target_col = None
        rf_classification_target_col = None

        for c in df.columns:
            try:
                nun = df[c].dropna().nunique()
                if nun == 2:
                    logistic_target_col = c
                    rf_classification_target_col = c
                    break
            except Exception:
                pass

        anova_target_col = numeric_cols[0] if len(numeric_cols) >= 1 else None
        anova_group_col = None
        for c in categorical_cols:
            try:
                nun = df[c].dropna().nunique()
                if 2 <= nun <= 10:
                    anova_group_col = c
                    break
            except Exception:
                pass

        manova_group_col = anova_group_col
        permanova_group_col = anova_group_col
        dependent_cols = numeric_cols[:3] if len(numeric_cols) >= 2 else None

        # =========================
        # Perfil
        # =========================
        profile = build_multivariate_profile(df)

        # =========================
        # Ejecutar análisis centralizado
        # =========================
        analysis_results, cache_rebuilt = get_or_build_multivariate_analysis_cache(
            db=db,
            ds=ds,
            df=df,
            dataset_id=dataset_id,
            plots_dir=PLOTS_DIR,
            research_type=research_type,
            regression_target_col=regression_target_col,
            rf_target_col=rf_target_col,
            logistic_target_col=logistic_target_col,
            rf_classification_target_col=rf_classification_target_col,
            anova_target_col=anova_target_col,
            anova_group_col=anova_group_col,
            manova_dependent_cols=dependent_cols,
            manova_group_col=manova_group_col,
            permanova_group_col=permanova_group_col,
        )

        corr_result = analysis_results.get("corr")
        pca_result = analysis_results.get("pca")
        cluster_result = analysis_results.get("cluster")
        efa_result = analysis_results.get("efa")
        regression_result = analysis_results.get("regression")
        rf_result = analysis_results.get("rf_regression")
        logistic_result = analysis_results.get("logistic")
        rf_classification_result = analysis_results.get("rf_classification")
        anova_result = analysis_results.get("anova")
        manova_result = analysis_results.get("manova")
        permanova_result = analysis_results.get("permanova")
        analysis_errors = analysis_results.get("errors", {})

        # =========================
        # Comparación de modelos
        # =========================
        model_comparison = None
        if regression_result or rf_result:
            linear = regression_result or {}
            rf = rf_result or {}

            winner = None
            try:
                lr_r2 = float(linear.get("r2")) if linear.get("r2") is not None else None
                rf_r2 = float(rf.get("r2")) if rf.get("r2") is not None else None

                if lr_r2 is not None and rf_r2 is not None:
                    if abs(lr_r2 - rf_r2) < 1e-6:
                        winner = "tie"
                    elif rf_r2 > lr_r2:
                        winner = "random_forest"
                    else:
                        winner = "linear_regression"
            except Exception:
                winner = None

            summary = []
            if winner == "random_forest":
                summary.append("Random Forest mostró mejor desempeño predictivo que la regresión lineal.")
            elif winner == "linear_regression":
                summary.append("La regresión lineal mostró mejor desempeño predictivo que Random Forest.")
            elif winner == "tie":
                summary.append("Ambos modelos mostraron un rendimiento muy similar.")

            model_comparison = {
                "available": True,
                "target_col": regression_target_col or rf_target_col,
                "linear": linear,
                "random_forest": rf,
                "winner": winner,
                "summary": summary,
            }

        # =========================
        # Visualizaciones por grupos
        # =========================
        group_visuals = None
        try:
            if anova_group_col and "generate_group_visualizations" in globals():
                group_visuals = generate_group_visualizations(
                    df=df,
                    group_col=anova_group_col,
                    dataset_id=dataset_id,
                    plots_dir=PLOTS_DIR,
                )
        except Exception as e:
            current_app.logger.warning(f"[group_visuals] ds{dataset_id}: {e}")
            group_visuals = {}

        # ===== Catálogo unificado de figuras =====
        figure_catalog = build_unified_multivariate_figure_catalog(
            dataset_id=dataset_id,
            plots_dir=PLOTS_DIR,
            group_visuals=group_visuals,
        )
        # ===== Ranking automático de insights =====
        try:
            insights_ranking = generate_insights_ranking(
                corr_result=corr_result,
                pca_result=pca_result,
                cluster_result=cluster_result,
                efa_result=efa_result,
                regression_result=regression_result,
                rf_result=rf_result,
                logistic_result=logistic_result,
                rf_classification_result=rf_classification_result,
                anova_result=anova_result,
                manova_result=manova_result,
                permanova_result=permanova_result,
                research_type=research_type,
                top_n=8,
            )
        except Exception as e:
            current_app.logger.warning(f"[multivariate_article:insights] ds{dataset_id}: {e}")
        # =========================
        # 🧠 CONCLUSIÓN AUTOMÁTICA SADI
        # =========================
        try:
            conclusion = generate_sadi_conclusion(
                profile=profile,
                corr_result=corr_result,
                pca_result=pca_result,
                cluster_result=cluster_result,
                efa_result=efa_result,
                regression_result=regression_result,
                rf_result=rf_result,
                logistic_result=logistic_result,
                anova_result=anova_result,
                manova_result=manova_result,
                permanova_result=permanova_result,
                research_type=research_type,
            )
        except Exception as e:
            current_app.logger.warning(f"[multivariate_report_word:conclusion] ds{dataset_id}: {e}")
            conclusion = None

        # =========================
        # Preparar estructuras auxiliares
        # =========================
        pca_summary = pca_result.get("summary") if pca_result else None
        pca_loadings = pca_result.get("loadings") if pca_result else None

        cluster_counts = cluster_result.get("cluster_counts") if cluster_result else None
        cluster_best_k = cluster_result.get("best_k") if cluster_result else None

        efa_kmo = efa_result.get("kmo") if efa_result else None
        efa_bartlett_p = efa_result.get("bartlett_p") if efa_result else None
        efa_n_factors = efa_result.get("n_factors") if efa_result else None
        efa_loadings = efa_result.get("loadings") if efa_result else None

        conclusion = generate_sadi_conclusion(
            profile=profile,
            corr_result=corr_result,
            pca_result=pca_result,
            cluster_result=cluster_result,
            efa_result=efa_result,
            regression_result=regression_result,
            rf_result=rf_result,
            logistic_result=logistic_result,
            anova_result=anova_result,
            manova_result=manova_result,
            permanova_result=permanova_result,
            research_type=research_type,
        )
        insights_ranking = generate_insights_ranking(
            corr_result=corr_result,
            pca_result=pca_result,
            cluster_result=cluster_result,
            efa_result=efa_result,
            regression_result=regression_result,
            rf_result=rf_result,
            logistic_result=logistic_result,
            rf_classification_result=rf_classification_result,
            anova_result=anova_result,
            manova_result=manova_result,
            permanova_result=permanova_result,
            research_type=research_type,
            top_n=8,
        )
        return render_template(
            "detail.html",
            ds=ds,
            df_preview=df.head(25).to_html(classes="table table-striped table-sm", index=False),

            profile=profile,
            research_type=research_type,

            corr_result=corr_result,
            pca_result=pca_result,
            cluster_result=cluster_result,
            efa_result=efa_result,

            regression_result=regression_result,
            rf_result=rf_result,
            logistic_result=logistic_result,
            rf_classification_result=rf_classification_result,

            anova_result=anova_result,
            manova_result=manova_result,
            permanova_result=permanova_result,

            model_comparison=model_comparison,
            group_visuals=group_visuals,

            plot_catalog=figure_catalog,

            pca_summary=pca_summary,
            pca_loadings=pca_loadings,
            cluster_counts=cluster_counts,
            cluster_best_k=cluster_best_k,
            efa_kmo=efa_kmo,
            efa_bartlett_p=efa_bartlett_p,
            efa_n_factors=efa_n_factors,
            efa_loadings=efa_loadings,

            analysis_errors=analysis_errors,
            cache_rebuilt=cache_rebuilt,
            cache_available=bool(getattr(ds, "analysis_cache", None)),
            conclusion=conclusion,
            insights_ranking=insights_ranking,
        )

    except Exception as e:
        current_app.logger.exception(f"[multivariate_detail] ds{dataset_id}: {e}")
        flash(f"No se pudo abrir el detalle multivariado: {e}", "danger")
        return redirect(url_for("dataset.dashboard"))
    finally:
        db.close()

@multivariate_bp.post("/<int:dataset_id>/recompute")
@login_required
def recompute_analysis_cache(dataset_id: int):
    db = SessionLocal()
    try:
        ds = db.get(Dataset, dataset_id)
        if not ds or ds.user_id != current_user.id:
            flash("Dataset no encontrado.", "warning")
            return redirect(url_for("dataset.dashboard"))

        ds.analysis_cache = None
        db.add(ds)
        db.commit()

        flash("Cache de análisis limpiado. Se recalculará al volver a abrir el análisis.", "success")
        return redirect(url_for("multivariate.multivariate_detail", dataset_id=dataset_id))
    except Exception as e:
        db.rollback()
        current_app.logger.exception(f"[recompute_analysis_cache] ds{dataset_id}: {e}")
        flash(f"No se pudo reiniciar el cache del análisis: {e}", "danger")
        return redirect(url_for("multivariate.multivariate_detail", dataset_id=dataset_id))
    finally:
        db.close()

@multivariate_bp.post("/<int:dataset_id>/corr")
@login_required
def multivariate_corr(dataset_id: int):
    db = SessionLocal()
    try:
        ds = db.get(Dataset, dataset_id)
        if not ds or ds.user_id != current_user.id:
            flash("Dataset no encontrado.", "warning")
            return redirect(url_for("dataset.dashboard"))

        path = os.path.join(UPLOAD_DIR, ds.filename)
        df = read_dataframe(path, ds.delimiter)

        corr_result = generate_correlation_heatmap(
            df=df,
            dataset_id=dataset_id,
            plots_dir=PLOTS_DIR,
            research_type=getattr(ds, "research_type", "general"),
        )

        flash(
            f"Heatmap generado correctamente con {corr_result['n_variables']} variables numéricas. "
            f"{corr_result.get('interpretacion', '')}",
            "success"
        )
        return redirect(url_for("multivariate.multivariate_detail", dataset_id=dataset_id))
    except Exception as e:
        current_app.logger.exception(f"[multivariate_corr] ds{dataset_id}: {e}")
        flash(f"No se pudo generar el heatmap de correlación: {e}", "danger")
        return redirect(url_for("multivariate.multivariate_detail", dataset_id=dataset_id))
    finally:
        db.close()


@multivariate_bp.post("/<int:dataset_id>/pca")
@login_required
def multivariate_pca(dataset_id: int):
    db = SessionLocal()
    try:
        ds = db.get(Dataset, dataset_id)
        if not ds or ds.user_id != current_user.id:
            flash("Dataset no encontrado.", "warning")
            return redirect(url_for("dataset.dashboard"))

        path = os.path.join(UPLOAD_DIR, ds.filename)
        df = read_dataframe(path, ds.delimiter)

        result = run_pca_analysis(
            df=df,
            dataset_id=dataset_id,
            plots_dir=PLOTS_DIR,
            research_type=getattr(ds, "research_type", "general"),
        )

        flash(
            f"PCA ejecutado correctamente con {result['n_variables']} variables numéricas. "
            f"{result.get('interpretacion', '')}",
            "success"
        )
        return redirect(url_for("multivariate.multivariate_detail", dataset_id=dataset_id))
    except Exception as e:
        current_app.logger.exception(f"[multivariate_pca] ds{dataset_id}: {e}")
        flash(f"No se pudo ejecutar PCA: {e}", "danger")
        return redirect(url_for("multivariate.multivariate_detail", dataset_id=dataset_id))
    finally:
        db.close()


@multivariate_bp.post("/<int:dataset_id>/cluster")
@login_required
def multivariate_cluster(dataset_id: int):
    db = SessionLocal()
    try:
        ds = db.get(Dataset, dataset_id)
        if not ds or ds.user_id != current_user.id:
            flash("Dataset no encontrado.", "warning")
            return redirect(url_for("dataset.dashboard"))

        path = os.path.join(UPLOAD_DIR, ds.filename)
        df = read_dataframe(path, ds.delimiter)

        result = run_kmeans_analysis(
        df=df,
        dataset_id=dataset_id,
        plots_dir=PLOTS_DIR,
        research_type=getattr(ds, "research_type", "general"),
    )

        flash(
            f"Clustering K-Means ejecutado correctamente. "
            f"SADI sugiere k={result['best_k']}. {result.get('interpretacion', '')}",
            "success"
        )
        return redirect(url_for("multivariate.multivariate_detail", dataset_id=dataset_id))
    except Exception as e:
        current_app.logger.exception(f"[multivariate_cluster] ds{dataset_id}: {e}")
        flash(f"No se pudo ejecutar clustering: {e}", "danger")
        return redirect(url_for("multivariate.multivariate_detail", dataset_id=dataset_id))
    finally:
        db.close()

@multivariate_bp.get("/<int:dataset_id>/report.pdf")
@login_required
def multivariate_report_pdf(dataset_id: int):
    db = SessionLocal()
    try:
        ds = db.get(Dataset, dataset_id)
        if not ds or ds.user_id != current_user.id:
            flash("Dataset no encontrado.", "warning")
            return redirect(url_for("dataset.dashboard"))

        current_kind = (getattr(ds, "dataset_type", None) or "").strip()
        if current_kind != "multivariate":
            flash("Este dataset no está marcado como multivariate.", "warning")
            return redirect(url_for("dataset.dataset_detail", dataset_id=dataset_id))

        path = os.path.join(UPLOAD_DIR, ds.filename)
        df = read_dataframe(path, ds.delimiter)

        profile = build_multivariate_profile(df)
        # ===== Tipo de investigación =====
        research_type = getattr(ds, "research_type", "general")

        # ===== Correlación =====
        corr_result = None
        try:
            corr_result = generate_correlation_heatmap(
                df=df,
                dataset_id=dataset_id,
                plots_dir=PLOTS_DIR,
                research_type=research_type,
            )
        except Exception as e:
            current_app.logger.warning(f"[multivariate_report_word:corr] ds{dataset_id}: {e}")
        pca_summary = None
        pca_loadings = None
        cluster_counts = None
        cluster_best_k = None

        try:
            pca_result = run_pca_analysis(
            df=df,
            dataset_id=dataset_id,
            plots_dir=PLOTS_DIR,
            research_type=research_type,
        )
            pca_summary = pca_result.get("summary_rows")
            pca_loadings = pca_result.get("loadings_table")
        except Exception as e:
            current_app.logger.warning(f"[multivariate_report_pdf:pca] ds{dataset_id}: {e}")

        try:
            pca_result = run_pca_analysis(
            df=df,
            dataset_id=dataset_id,
            plots_dir=PLOTS_DIR,
            research_type=research_type,
        )
            cluster_counts = cluster_result.get("cluster_counts")
            cluster_best_k = cluster_result.get("best_k")
        except Exception as e:
            current_app.logger.warning(f"[multivariate_report_pdf:cluster] ds{dataset_id}: {e}")
        out_name = f"ds{dataset_id}_multivariate_report.pdf"
        out_path = os.path.join(PLOTS_DIR, out_name)

        efa_kmo = None
        efa_bartlett_p = None
        efa_n_factors = None
        efa_loadings = None

        try:
            efa_result = run_efa_analysis(
                df=df,
                dataset_id=dataset_id,
                plots_dir=PLOTS_DIR,
            )
            efa_kmo = efa_result.get("kmo")
            efa_bartlett_p = efa_result.get("bartlett_p")
            efa_n_factors = efa_result.get("n_factors")
            efa_loadings = efa_result.get("loadings_table")
        except Exception as e:
            current_app.logger.warning(f"[multivariate_report_pdf:efa] ds{dataset_id}: {e}")

        generate_multivariate_report_pdf(
            dataset_title=ds.title or ds.filename,
            profile=profile,
            plots_dir=PLOTS_DIR,
            dataset_id=dataset_id,
            output_path=out_path,
            pca_summary=pca_summary,
            pca_loadings=pca_loadings,
            cluster_counts=cluster_counts,
            cluster_best_k=cluster_best_k,
            efa_kmo=efa_kmo,
            efa_bartlett_p=efa_bartlett_p,
            efa_n_factors=efa_n_factors,
            efa_loadings=efa_loadings,
        )

        if not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
            flash("No se pudo generar correctamente el PDF multivariado.", "danger")
            return redirect(url_for("multivariate.multivariate_detail", dataset_id=dataset_id))

        return send_file(
            out_path,
            as_attachment=True,
            download_name=out_name,
            mimetype="application/pdf",
        )

    except Exception as e:
        current_app.logger.exception(f"[multivariate_report_pdf] ds{dataset_id}: {e}")
        flash(f"No se pudo generar el PDF multivariado: {e}", "danger")
        return redirect(url_for("multivariate.multivariate_detail", dataset_id=dataset_id))
    finally:
        db.close()

@multivariate_bp.get("/<int:dataset_id>/report.docx")
@login_required
def multivariate_report_word(dataset_id: int):
    db = SessionLocal()
    try:
        ds = db.get(Dataset, dataset_id)
        if not ds or ds.user_id != current_user.id:
            flash("Dataset no encontrado.", "warning")
            return redirect(url_for("dataset.dashboard"))

        # =========================
        # SADI: detectar tipo
        # =========================
        current_kind = (
            getattr(ds, "dataset_type", None)
            or getattr(ds, "dataset_kind", None)
            or "dataset"
        )
        current_kind = str(current_kind).strip()

        if current_kind == "multivariate":
            mode = "multivariate"
        elif current_kind in ("survey_likert", "survey_normal"):
            mode = "survey"
        else:
            mode = "general"

        current_app.logger.info(
            f"[SADI] ds{dataset_id}: exportando en modo '{mode}' (tipo real: '{current_kind}')"
        )

        research_type = getattr(ds, "research_type", "general")

        path = os.path.join(UPLOAD_DIR, ds.filename)
        df = read_dataframe(path, ds.delimiter)

        # =========================
        # PERFIL BASE (siempre)
        # =========================

        analysis_meta = analyze_dataset_with_recommendations(
            df,
            dataset_type=current_kind or "dataset",
            research_area=getattr(ds, "research_area", None) or "general",
        ) or {}

        quick_recommendations = analysis_meta.get("quick_recommendations", []) or []
        suggested_plan = analysis_meta.get("suggested_plan", {}) or {}
        insights = analysis_meta.get("insights", []) or []
        insights_text = analysis_meta.get("insights_text")
        plot_catalog_general = build_general_dataset_figure_catalog(
            dataset_id=dataset_id,
            plots_dir=PLOTS_DIR,
        )
        plot_summary = summarize_plot_tags(
            [item["basename"] for item in plot_catalog_general]
        )

        profile = build_multivariate_profile(df)

        # =========================
        # INICIALIZACIÓN GLOBAL
        # =========================
        corr_result = None
        pca_result = None
        cluster_result = None
        efa_result = None

        regression_result = None
        rf_result = None
        logistic_result = None
        rf_classification_result = None

        anova_result = None
        manova_result = None
        permanova_result = None

        group_visuals = {}
        figure_catalog = []
        insights_ranking = []
        conclusion = None
        abstract_text = None
        keywords = None

        out_name = f"ds{dataset_id}_report.docx"
        out_path = os.path.join(PLOTS_DIR, out_name)

        # =====================================================
        # SADI CORE: lógica adaptable por tipo
        # =====================================================

        # ===== GENERAL =====
        try:
            corr_result = generate_correlation_heatmap(
                df=df,
                dataset_id=dataset_id,
                plots_dir=PLOTS_DIR,
                research_type=research_type,
            )
        except Exception as e:
            current_app.logger.warning(f"[corr] {e}")

        # ===== SOLO SI ES MULTIVARIANTE =====
        if mode == "multivariate":

            try:
                pca_result = run_pca_analysis(
                    df=df,
                    dataset_id=dataset_id,
                    plots_dir=PLOTS_DIR,
                    research_type=research_type,
                )
            except Exception as e:
                current_app.logger.warning(f"[pca] {e}")

            try:
                cluster_result = run_kmeans_analysis(
                    df=df,
                    dataset_id=dataset_id,
                    plots_dir=PLOTS_DIR,
                    research_type=research_type,
                )
            except Exception as e:
                current_app.logger.warning(f"[cluster] {e}")

            try:
                efa_result = run_efa_analysis(
                    df=df,
                    dataset_id=dataset_id,
                    plots_dir=PLOTS_DIR,
                )
            except Exception as e:
                current_app.logger.warning(f"[efa] {e}")

        # ===== MODELOS (solo si hay numéricas suficientes) =====
        numeric_columns = df.select_dtypes(include="number").columns.tolist()

        if len(numeric_columns) >= 2:
            target_col = numeric_columns[-1]

            try:
                regression_result = run_regression_analysis(
                    df=df,
                    dataset_id=dataset_id,
                    plots_dir=PLOTS_DIR,
                    target_col=target_col,
                    research_type=research_type,
                )
            except Exception as e:
                current_app.logger.warning(f"[regression] {e}")

            try:
                rf_result = run_rf_regression_analysis(
                    df=df,
                    dataset_id=dataset_id,
                    plots_dir=PLOTS_DIR,
                    target_col=target_col,
                    research_type=research_type,
                )
            except Exception as e:
                current_app.logger.warning(f"[rf] {e}")

        # ===== FIGURAS =====
        try:
            figure_catalog = build_unified_multivariate_figure_catalog(
                dataset_id=dataset_id,
                plots_dir=PLOTS_DIR,
                group_visuals=group_visuals,
            )
        except Exception as e:
            current_app.logger.warning(f"[figures] {e}")

        # ===== INSIGHTS =====
        try:
            insights_ranking = generate_insights_ranking(
                corr_result=corr_result,
                pca_result=pca_result,
                cluster_result=cluster_result,
                efa_result=efa_result,
                regression_result=regression_result,
                rf_result=rf_result,
                research_type=research_type,
                top_n=6,
            )
        except Exception as e:
            current_app.logger.warning(f"[insights] {e}")

        # ===== CONCLUSIÓN =====
        try:
            conclusion = generate_sadi_conclusion(
                profile=profile,
                corr_result=corr_result,
                pca_result=pca_result,
                cluster_result=cluster_result,
                efa_result=efa_result,
                regression_result=regression_result,
                rf_result=rf_result,
                research_type=research_type,
            )
        except Exception as e:
            current_app.logger.warning(f"[conclusion] {e}")

        # ===== ABSTRACT =====
        try:
            abstract_data = generate_sadi_abstract_and_keywords(
                profile=profile,
                corr_result=corr_result,
                pca_result=pca_result,
                cluster_result=cluster_result,
                efa_result=efa_result,
                regression_result=regression_result,
                rf_result=rf_result,
                research_type=research_type,
            )

            if isinstance(abstract_data, dict):
                abstract_text = abstract_data.get("abstract")
                keywords = abstract_data.get("keywords")

        except Exception as e:
            current_app.logger.warning(f"[abstract] {e}")

        # =====================================================
        # GENERACIÓN FINAL
        # =====================================================
        generate_multivariate_report_docx(
            dataset_title=ds.title,
            profile=profile,
            plots_dir=PLOTS_DIR,
            dataset_id=dataset_id,
            output_path=out_path,

            figure_catalog=figure_catalog,

            pca_summary=(pca_result or {}).get("summary"),
            pca_loadings=(pca_result or {}).get("loadings"),

            cluster_counts=(cluster_result or {}).get("cluster_counts"),
            cluster_best_k=(cluster_result or {}).get("best_k"),

            efa_kmo=(efa_result or {}).get("kmo") if efa_result else None,
            efa_bartlett_p=(efa_result or {}).get("bartlett_p") if efa_result else None,
            efa_n_factors=(efa_result or {}).get("n_factors") if efa_result else None,
            efa_loadings=(efa_result or {}).get("loadings") if efa_result else None,

            regression_result=regression_result,
            rf_result=rf_result,

            conclusion=conclusion,
            abstract_text=abstract_text,
            keywords=keywords,
            insights_ranking=insights_ranking,
            analysis_meta=analysis_meta,
            insights=insights,
            insights_text=insights_text,
            quick_recommendations=quick_recommendations,
            suggested_plan=suggested_plan,
            plot_summary=plot_summary,
            general_figure_catalog=plot_catalog_general,
        )

        if not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
            flash("No se pudo generar correctamente el Word.", "danger")
            return redirect(url_for("dataset.dataset_detail", dataset_id=dataset_id))

        return send_file(
            out_path,
            as_attachment=True,
            download_name=out_name,
            mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

    except Exception as e:
        current_app.logger.exception(f"[multivariate_report_word] ds{dataset_id}: {e}")
        flash(f"No se pudo generar el Word: {e}", "danger")
        return redirect(url_for("dataset.dataset_detail", dataset_id=dataset_id))

    finally:
        db.close()


@multivariate_bp.get("/<int:dataset_id>/article.docx")
@login_required
def multivariate_article(dataset_id: int):
    db = SessionLocal()

    try:
        ds = db.get(Dataset, dataset_id)

        if not ds or ds.user_id != current_user.id:
            flash("Dataset no encontrado.", "warning")
            return redirect(url_for("dataset.dashboard"))

        # =========================
        # SADI: detectar tipo
        # =========================
        current_kind = (
            getattr(ds, "dataset_type", None)
            or getattr(ds, "dataset_kind", None)
            or "dataset"
        )
        current_kind = str(current_kind).strip()

        if current_kind == "multivariate":
            mode = "multivariate"
        elif current_kind in ("survey_likert", "survey_normal"):
            mode = "survey"
        else:
            mode = "general"

        current_app.logger.info(
            f"[SADI ARTICLE] ds{dataset_id}: modo '{mode}' (tipo: '{current_kind}')"
        )

        research_type = getattr(ds, "research_type", "general")

        path = os.path.join(UPLOAD_DIR, ds.filename)
        df = read_dataframe(path, ds.delimiter)

        analysis_meta = analyze_dataset_with_recommendations(
            df,
            dataset_type=current_kind or "dataset",
            research_area=getattr(ds, "research_area", None) or "general",
        ) or {}

        quick_recommendations = analysis_meta.get("quick_recommendations", []) or []
        suggested_plan = analysis_meta.get("suggested_plan", {}) or {}
        insights = analysis_meta.get("insights", []) or []
        insights_text = analysis_meta.get("insights_text")
        plot_catalog_general = build_general_dataset_figure_catalog(
            dataset_id=dataset_id,
            plots_dir=PLOTS_DIR,
        )
        plot_summary = summarize_plot_tags(
            [item["basename"] for item in plot_catalog_general]
        )

        profile = build_multivariate_profile(df)

        out_name = f"ds{dataset_id}_article.docx"
        out_path = os.path.join(PLOTS_DIR, out_name)

        # =========================
        # INICIALIZACIÓN
        # =========================
        corr_result = None
        pca_result = None
        cluster_result = None
        efa_result = None

        regression_result = None
        rf_result = None
        logistic_result = None
        rf_classification_result = None

        anova_result = None
        manova_result = None
        permanova_result = None

        group_visuals = {}
        figure_catalog = []
        insights_ranking = []
        interpretacion = None
        conclusion = None
        abstract_text = None
        keywords = None
        limitations = None
        future_work = None

        # =========================
        # CORE SADI
        # =========================

        # Siempre
        try:
            corr_result = generate_correlation_heatmap(
                df=df,
                dataset_id=dataset_id,
                plots_dir=PLOTS_DIR,
                research_type=research_type,
            )
        except Exception as e:
            current_app.logger.warning(f"[corr] {e}")

        # SOLO multivariate
        if mode == "multivariate":

            try:
                pca_result = run_pca_analysis(
                    df=df,
                    dataset_id=dataset_id,
                    plots_dir=PLOTS_DIR,
                    research_type=research_type,
                )
            except Exception as e:
                current_app.logger.warning(f"[pca] {e}")

            try:
                cluster_result = run_kmeans_analysis(
                    df=df,
                    dataset_id=dataset_id,
                    plots_dir=PLOTS_DIR,
                    research_type=research_type,
                )
            except Exception as e:
                current_app.logger.warning(f"[cluster] {e}")

            try:
                efa_result = run_efa_analysis(
                    df=df,
                    dataset_id=dataset_id,
                    plots_dir=PLOTS_DIR,
                )
            except Exception as e:
                current_app.logger.warning(f"[efa] {e}")

        # Modelos básicos si hay numéricas
        numeric_columns = df.select_dtypes(include="number").columns.tolist()

        if len(numeric_columns) >= 2:
            target_col = numeric_columns[-1]

            try:
                regression_result = run_regression_analysis(
                    df=df,
                    dataset_id=dataset_id,
                    plots_dir=PLOTS_DIR,
                    target_col=target_col,
                    research_type=research_type,
                )
            except Exception as e:
                current_app.logger.warning(f"[regression] {e}")

            try:
                rf_result = run_rf_regression_analysis(
                    df=df,
                    dataset_id=dataset_id,
                    plots_dir=PLOTS_DIR,
                    target_col=target_col,
                    research_type=research_type,
                )
            except Exception as e:
                current_app.logger.warning(f"[rf] {e}")

        # =========================
        # INTERPRETACIÓN
        # =========================
        try:
            interpretacion = generate_multivariate_interpretation(
                pca_loadings=(pca_result or {}).get("loadings"),
                efa_loadings=(efa_result or {}).get("loadings"),
                cluster_counts=(cluster_result or {}).get("cluster_counts"),
            )
        except Exception as e:
            current_app.logger.warning(f"[interpretacion] {e}")

        # =========================
        # FIGURAS
        # =========================
        try:
            figure_catalog = build_unified_multivariate_figure_catalog(
                dataset_id=dataset_id,
                plots_dir=PLOTS_DIR,
                group_visuals=group_visuals,
            )
        except Exception as e:
            current_app.logger.warning(f"[figures] {e}")

        # =========================
        # INSIGHTS + CONCLUSIÓN
        # =========================
        try:
            insights_ranking = generate_insights_ranking(
                corr_result=corr_result,
                pca_result=pca_result,
                cluster_result=cluster_result,
                efa_result=efa_result,
                regression_result=regression_result,
                rf_result=rf_result,
                research_type=research_type,
                top_n=8,
            )
        except Exception as e:
            current_app.logger.warning(f"[insights] {e}")

        try:
            conclusion = generate_sadi_conclusion(
                profile=profile,
                corr_result=corr_result,
                pca_result=pca_result,
                cluster_result=cluster_result,
                efa_result=efa_result,
                regression_result=regression_result,
                rf_result=rf_result,
                research_type=research_type,
            )
        except Exception as e:
            current_app.logger.warning(f"[conclusion] {e}")

        # =========================
        # ABSTRACT
        # =========================
        try:
            abstract_data = generate_sadi_abstract_and_keywords(
                profile=profile,
                corr_result=corr_result,
                pca_result=pca_result,
                cluster_result=cluster_result,
                efa_result=efa_result,
                regression_result=regression_result,
                rf_result=rf_result,
                research_type=research_type,
            )

            if isinstance(abstract_data, dict):
                abstract_text = abstract_data.get("abstract")
                keywords = abstract_data.get("keywords")

        except Exception as e:
            current_app.logger.warning(f"[abstract] {e}")

        # =========================
        # GENERACIÓN DOCX
        # =========================
        generate_multivariate_article_docx(
            dataset_title=ds.title or ds.filename,
            profile=profile,
            plots_dir=PLOTS_DIR,
            dataset_id=dataset_id,
            output_path=out_path,
            figure_catalog=figure_catalog,

            pca_summary=(pca_result or {}).get("summary"),
            pca_loadings=(pca_result or {}).get("loadings"),

            cluster_counts=(cluster_result or {}).get("cluster_counts"),
            cluster_best_k=(cluster_result or {}).get("best_k"),

            efa_kmo=(efa_result or {}).get("kmo") if efa_result else None,
            efa_bartlett_p=(efa_result or {}).get("bartlett_p") if efa_result else None,
            efa_n_factors=(efa_result or {}).get("n_factors") if efa_result else None,
            efa_loadings=(efa_result or {}).get("loadings") if efa_result else None,

            regression_result=regression_result,
            rf_result=rf_result,

            interpretacion=interpretacion,
            abstract_text=abstract_text,
            keywords=keywords,
            conclusion=conclusion,
            insights_ranking=insights_ranking,
            analysis_meta=analysis_meta,
            insights=insights,
            insights_text=insights_text,
            quick_recommendations=quick_recommendations,
            suggested_plan=suggested_plan,
            plot_summary=plot_summary,
            general_figure_catalog=plot_catalog_general,
        )

        if not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
            flash("No se pudo generar correctamente el artículo.", "danger")
            return redirect(url_for("dataset.dataset_detail", dataset_id=dataset_id))

        return send_file(
            out_path,
            as_attachment=True,
            download_name=out_name,
            mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

    except Exception as e:
        current_app.logger.exception(f"[multivariate_article] ds{dataset_id}: {e}")
        flash(f"No se pudo generar el artículo: {e}", "danger")
        return redirect(url_for("dataset.dataset_detail", dataset_id=dataset_id))

    finally:
        db.close()

@multivariate_bp.post("/<int:dataset_id>/efa")
@login_required
def multivariate_efa(dataset_id: int):
    db = SessionLocal()
    try:
        ds = db.get(Dataset, dataset_id)
        if not ds or ds.user_id != current_user.id:
            flash("Dataset no encontrado.", "warning")
            return redirect(url_for("dataset.dashboard"))

        path = os.path.join(UPLOAD_DIR, ds.filename)
        df = read_dataframe(path, ds.delimiter)

        result = run_efa_analysis(
            df=df,
            dataset_id=dataset_id,
            plots_dir=PLOTS_DIR,
        )

        flash(
            f"EFA ejecutado correctamente. Factores sugeridos: {result.get('n_factors', '—')}.",
            "success"
        )
        return redirect(url_for("multivariate.multivariate_detail", dataset_id=dataset_id))

    except Exception as e:
        current_app.logger.exception(f"[efa] ds{dataset_id}: {e}")

        msg = str(e)
        if "Singular matrix" in msg:
            flash(
                "No se pudo ejecutar EFA porque la matriz de correlación es singular. "
                "Esto suele ocurrir cuando hay variables duplicadas, correlaciones extremas o poca variabilidad."
            )
        else:
            flash(f"No se pudo ejecutar EFA: {e}")

        return redirect(url_for("multivariate.multivariate_detail", dataset_id=dataset_id))
    finally:
        db.close()

@multivariate_bp.post("/<int:dataset_id>/regression")
@login_required
def multivariate_regression(dataset_id: int):
    db = SessionLocal()
    try:
        ds = db.get(Dataset, dataset_id)
        if not ds or ds.user_id != current_user.id:
            flash("Dataset no encontrado.", "warning")
            return redirect(url_for("dataset.dashboard"))

        path = os.path.join(UPLOAD_DIR, ds.filename)
        df = read_dataframe(path, ds.delimiter)

        num_cols = list(df.select_dtypes(include=["number"]).columns)
        if len(num_cols) < 3:
            flash("No hay suficientes variables numéricas para regresión.", "danger")
            return redirect(url_for("multivariate.multivariate_detail", dataset_id=dataset_id))

        target_col = (request.form.get("target_col") or "").strip()
        if not target_col:
            target_col = num_cols[-1]

        if target_col not in num_cols:
            flash("La variable objetivo seleccionada no es numérica válida para regresión.", "warning")
            return redirect(url_for("multivariate.multivariate_detail", dataset_id=dataset_id))

        regression_result = run_regression_analysis(
            df=df,
            dataset_id=dataset_id,
            plots_dir=PLOTS_DIR,
            target_col=target_col,
        )

        # =========================
        # Guardar en manifest_data
        # =========================
        manifest_data = read_manifest_data(dataset_id) or {}
        if not isinstance(manifest_data, dict):
            manifest_data = {}

        model_results = manifest_data.get("model_results", {})
        if not isinstance(model_results, dict):
            model_results = {}

        model_results["regression_result"] = regression_result
        manifest_data["model_results"] = model_results

        write_manifest(dataset_id, manifest_data)

        # DEBUG inmediato
        manifest_check = read_manifest_data(dataset_id) or {}
        current_app.logger.warning(
            f"[DEBUG regression saved] ds{dataset_id}: keys={list((manifest_check.get('model_results') or {}).keys())}"
        )

        # =========================
        # Guardar estado en sesión
        # =========================
        mv_state = session.get("mv_state", {})
        ds_key = str(dataset_id)
        mv_state.setdefault(ds_key, {})
        mv_state[ds_key]["regression_target_col"] = target_col
        session["mv_state"] = mv_state

        flash(
            f"Regresión ejecutada correctamente usando '{regression_result['target_col']}' como variable objetivo.",
            "success"
        )
        return redirect(url_for("multivariate.multivariate_detail", dataset_id=dataset_id))

    except Exception as e:
        current_app.logger.exception(f"[multivariate_regression] ds{dataset_id}: {e}")
        flash(f"No se pudo ejecutar la regresión: {e}", "danger")
        return redirect(url_for("multivariate.multivariate_detail", dataset_id=dataset_id))
    finally:
        db.close()

@multivariate_bp.post("/<int:dataset_id>/outliers")
@login_required
def multivariate_outliers(dataset_id: int):
    db = SessionLocal()
    try:
        ds = db.get(Dataset, dataset_id)
        if not ds or ds.user_id != current_user.id:
            flash("Dataset no encontrado.", "warning")
            return redirect(url_for("dataset.dashboard"))

        path = os.path.join(UPLOAD_DIR, ds.filename)
        df = read_dataframe(path, ds.delimiter)

        result = run_outlier_analysis(
            df=df,
            dataset_id=dataset_id,
            plots_dir=PLOTS_DIR,
        )

        flash(
            f"Detección de outliers completada. Posibles outliers encontrados: {result['outlier_count']}.",
            "success"
        )
        return redirect(url_for("multivariate.multivariate_detail", dataset_id=dataset_id))

    except Exception as e:
        current_app.logger.exception(f"[multivariate_outliers] ds{dataset_id}: {e}")
        flash(f"No se pudo ejecutar la detección de outliers: {e}", "danger")
        return redirect(url_for("multivariate.multivariate_detail", dataset_id=dataset_id))
    finally:
        db.close()

@multivariate_bp.post("/<int:dataset_id>/anova")
@login_required
def multivariate_anova(dataset_id: int):
    db = SessionLocal()
    try:
        ds = db.get(Dataset, dataset_id)
        if not ds or ds.user_id != current_user.id:
            flash("Dataset no encontrado.", "warning")
            return redirect(url_for("dataset.dashboard"))

        path = os.path.join(UPLOAD_DIR, ds.filename)
        df = read_dataframe(path, ds.delimiter)

        target_col = (request.form.get("anova_target_col") or "").strip()
        group_col = (request.form.get("anova_group_col") or "").strip()

        if not target_col or not group_col:
            flash("Debes seleccionar variable dependiente y variable de grupo para ANOVA.", "danger")
            return redirect(url_for("multivariate.multivariate_detail", dataset_id=dataset_id))

        result = run_anova_analysis(
            df=df,
            dataset_id=dataset_id,
            plots_dir=PLOTS_DIR,
            target_col=target_col,
            group_col=group_col,
        )

        mv_state = session.get("mv_state", {})
        ds_key = str(dataset_id)
        mv_state.setdefault(ds_key, {})
        mv_state[ds_key]["anova_target_col"] = target_col
        mv_state[ds_key]["anova_group_col"] = group_col
        session["mv_state"] = mv_state

        flash(
            f"ANOVA ejecutado correctamente con '{target_col}' como dependiente y '{group_col}' como grupo.",
            "success"
        )
        return redirect(url_for("multivariate.multivariate_detail", dataset_id=dataset_id))

    except Exception as e:
        current_app.logger.exception(f"[multivariate_anova] ds{dataset_id}: {e}")
        flash(f"No se pudo ejecutar ANOVA: {e}", "danger")
        return redirect(url_for("multivariate.multivariate_detail", dataset_id=dataset_id))
    finally:
        db.close()


@multivariate_bp.post("/<int:dataset_id>/manova")
@login_required
def multivariate_manova(dataset_id: int):
    db = SessionLocal()
    try:
        ds = db.get(Dataset, dataset_id)
        if not ds or ds.user_id != current_user.id:
            flash("Dataset no encontrado.", "warning")
            return redirect(url_for("dataset.dashboard"))

        path = os.path.join(UPLOAD_DIR, ds.filename)
        df = read_dataframe(path, ds.delimiter)

        group_col = (request.form.get("manova_group_col") or "").strip()
        if not group_col:
            flash("Debes seleccionar variable de grupo para MANOVA.", "danger")
            return redirect(url_for("multivariate.multivariate_detail", dataset_id=dataset_id))

        dependent_cols = [
            c for c in df.select_dtypes(include=["number"]).columns
            if c != group_col
        ]

        result = run_manova_analysis(
            df=df,
            dependent_cols=dependent_cols,
            group_col=group_col,
        )

        # guardar resultado en sesión simple? no. reconstruimos en detail por querystring
        mv_state = session.get("mv_state", {})
        ds_key = str(dataset_id)
        mv_state.setdefault(ds_key, {})
        mv_state[ds_key]["manova_group_col"] = group_col
        session["mv_state"] = mv_state

        flash(f"MANOVA ejecutado correctamente con '{group_col}' como grupo.", "success")
        return redirect(url_for("multivariate.multivariate_detail", dataset_id=dataset_id))

    except Exception as e:
        current_app.logger.exception(f"[multivariate_manova] ds{dataset_id}: {e}")
        flash(f"No se pudo ejecutar MANOVA: {e}", "danger")
        return redirect(url_for("multivariate.multivariate_detail", dataset_id=dataset_id))
    finally:
        db.close()


@multivariate_bp.post("/<int:dataset_id>/permanova")
@login_required
def multivariate_permanova(dataset_id: int):
    db = SessionLocal()
    try:
        ds = db.get(Dataset, dataset_id)
        if not ds or ds.user_id != current_user.id:
            flash("Dataset no encontrado.", "warning")
            return redirect(url_for("dataset.dashboard"))

        path = os.path.join(UPLOAD_DIR, ds.filename)
        df = read_dataframe(path, ds.delimiter)

        group_col = (request.form.get("permanova_group_col") or "").strip()
        if not group_col:
            flash("Debes seleccionar variable de grupo para PERMANOVA.", "danger")
            return redirect(url_for("multivariate.multivariate_detail", dataset_id=dataset_id))

        result = run_permanova_analysis(
            df=df,
            group_col=group_col,
            permutations=499,
        )

        mv_state = session.get("mv_state", {})
        ds_key = str(dataset_id)
        mv_state.setdefault(ds_key, {})
        mv_state[ds_key]["permanova_group_col"] = group_col
        session["mv_state"] = mv_state

        flash(
            f"PERMANOVA ejecutado correctamente con '{group_col}' como variable de grupo.",
            "success"
        )
        return redirect(url_for("multivariate.multivariate_detail", dataset_id=dataset_id))

    except Exception as e:
        current_app.logger.exception(f"[multivariate_permanova] ds{dataset_id}: {e}")
        flash(f"No se pudo ejecutar PERMANOVA: {e}", "danger")
        return redirect(url_for("multivariate.multivariate_detail", dataset_id=dataset_id))
    finally:
        db.close()

@multivariate_bp.post("/<int:dataset_id>/rf_regression")
@login_required
def multivariate_rf_regression(dataset_id: int):
    db = SessionLocal()
    try:
        ds = db.get(Dataset, dataset_id)
        if not ds or ds.user_id != current_user.id:
            flash("Dataset no encontrado.", "warning")
            return redirect(url_for("dataset.dashboard"))

        target_col = (request.form.get("target_col") or "").strip()
        if not target_col:
            flash("Debes seleccionar una variable objetivo para Random Forest.", "warning")
            return redirect(url_for("multivariate.multivariate_detail", dataset_id=dataset_id))

        path = os.path.join(UPLOAD_DIR, ds.filename)
        df = read_dataframe(path, ds.delimiter)

        rf_result = run_rf_regression_analysis(
            df=df,
            dataset_id=dataset_id,
            plots_dir=PLOTS_DIR,
            target_col=target_col,
        )

        # =========================
        # Guardar en manifest_data
        # =========================
        manifest_data = read_manifest_data(dataset_id) or {}
        if not isinstance(manifest_data, dict):
            manifest_data = {}

        model_results = manifest_data.get("model_results", {})
        if not isinstance(model_results, dict):
            model_results = {}

        model_results["rf_result"] = rf_result
        manifest_data["model_results"] = model_results

        write_manifest(dataset_id, manifest_data)

        # DEBUG inmediato
        manifest_check = read_manifest_data(dataset_id) or {}
        current_app.logger.warning(
            f"[DEBUG rf saved] ds{dataset_id}: keys={list((manifest_check.get('model_results') or {}).keys())}"
        )

        # =========================
        # Guardar estado en sesión
        # =========================
        mv_state = session.get("mv_state", {})
        ds_key = str(dataset_id)
        mv_state.setdefault(ds_key, {})
        mv_state[ds_key]["rf_target_col"] = target_col
        session["mv_state"] = mv_state

        flash(f"Random Forest ejecutado correctamente con '{target_col}' como objetivo.", "success")
        return redirect(url_for("multivariate.multivariate_detail", dataset_id=dataset_id))

    except Exception as e:
        current_app.logger.exception(f"[multivariate_rf_regression] ds{dataset_id}: {e}")
        flash(f"No se pudo ejecutar Random Forest: {e}", "danger")
        return redirect(url_for("multivariate.multivariate_detail", dataset_id=dataset_id))
    finally:
        db.close()
@multivariate_bp.post("/<int:dataset_id>/logistic")
@login_required
def multivariate_logistic(dataset_id: int):
    db = SessionLocal()
    try:
        ds = db.get(Dataset, dataset_id)

        target_col = (request.form.get("target_col") or "").strip()

        path = os.path.join(UPLOAD_DIR, ds.filename)
        df = read_dataframe(path, ds.delimiter)

        _ = run_logistic_regression_analysis(
            df=df,
            dataset_id=dataset_id,
            plots_dir=PLOTS_DIR,
            target_col=target_col,
        )

        mv_state = session.get("mv_state", {})
        mv_state.setdefault(str(dataset_id), {})
        mv_state[str(dataset_id)]["logistic_target_col"] = target_col
        session["mv_state"] = mv_state

        flash("Logistic Regression ejecutado correctamente.", "success")

    except Exception as e:
        flash(f"Error Logistic: {e}", "danger")

    return redirect(url_for("multivariate.multivariate_detail", dataset_id=dataset_id))
@multivariate_bp.post("/<int:dataset_id>/rf_classification")
@login_required
def multivariate_rf_classification(dataset_id: int):
    db = SessionLocal()
    try:
        ds = db.get(Dataset, dataset_id)
        if not ds or ds.user_id != current_user.id:
            flash("Dataset no encontrado.", "warning")
            return redirect(url_for("dataset.dashboard"))

        target_col = (request.form.get("target_col") or "").strip()
        if not target_col:
            flash("Debes seleccionar una variable objetivo.", "warning")
            return redirect(url_for("multivariate.multivariate_detail", dataset_id=dataset_id))

        path = os.path.join(UPLOAD_DIR, ds.filename)
        df = read_dataframe(path, ds.delimiter)

        result = run_rf_classification_analysis(
            df=df,
            dataset_id=dataset_id,
            plots_dir=PLOTS_DIR,
            target_col=target_col,
        )

        mv_state = session.get("mv_state", {})
        ds_key = str(dataset_id)
        mv_state.setdefault(ds_key, {})
        mv_state[ds_key]["rf_target_col"] = target_col
        session["mv_state"] = mv_state

        manifest_data = read_manifest_data(dataset_id) or {}
        if not isinstance(manifest_data, dict):
            manifest_data = {}

        model_results = manifest_data.get("model_results", {})
        if not isinstance(model_results, dict):
            model_results = {}

        model_results["rf_result"] = result
        manifest_data["model_results"] = model_results

        write_manifest(dataset_id, manifest_data)

        flash(f"Random Forest Clasificación ejecutado con '{target_col}'.", "success")
        return redirect(url_for("multivariate.multivariate_detail", dataset_id=dataset_id))

    except Exception as e:
        current_app.logger.exception(f"[rf_classification] ds{dataset_id}: {e}")
        flash(f"Error en clasificación: {e}", "danger")
        return redirect(url_for("multivariate.multivariate_detail", dataset_id=dataset_id))

    finally:
        db.close()