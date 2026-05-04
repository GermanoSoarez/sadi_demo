from __future__ import annotations

import os
import threading
import time
import webbrowser

from flask import Flask, redirect, session, url_for
from flask_login import current_user, login_user
from sqlalchemy import inspect, select

from config import Settings, STATIC_DIR, TEMPLATES_DIR, PLOTS_DIR, UPLOAD_DIR, MANIFESTS_DIR
import extensions as ext
from models import Base, User
from models import Dataset
# blueprints
from blueprints.auth.routes import auth_bp
from blueprints.dataset.routes import dataset_bp
from blueprints.likert.routes import likert_bp
from blueprints.survey.routes import survey_bp
from blueprints.multivariate.routes import multivariate_bp


# =========================================================
# CONFIG DEMO
# =========================================================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
INSTANCE_DIR = os.path.join(BASE_DIR, "instance")
os.makedirs(INSTANCE_DIR, exist_ok=True)

DEMO_DB_PATH = os.path.join(INSTANCE_DIR, "sadi_demo.db")
DEMO_DB_URL = f"sqlite:///{DEMO_DB_PATH}"

DEMO_USER_EMAIL = "demo@sadi.local"
DEMO_USER_NAME = "Dr. Campoy Demo"


# =========================================================
# RECONFIGURAR SQLALCHEMY A SQLITE PARA LA DEMO
# =========================================================
ext.reconfigure_database(DEMO_DB_URL)

SessionLocal = ext.SessionLocal
engine = ext.engine
login_manager = ext.login_manager


# =========================================================
# INICIALIZACIÓN DB
# =========================================================
def ensure_schema() -> None:
    insp = inspect(engine)
    if not insp.has_table("users"):
        Base.metadata.create_all(bind=engine)
        print("✅ Tablas demo creadas en SQLite")
    else:
        Base.metadata.create_all(bind=engine)
        print("✅ Esquema verificado en SQLite")


def ensure_demo_user() -> None:
    with SessionLocal() as db:
        exists = db.execute(
            select(User).where(User.email == DEMO_USER_EMAIL)
        ).scalar_one_or_none()

        if not exists:
            u = User(name=DEMO_USER_NAME, email=DEMO_USER_EMAIL)
            # dejamos password por compatibilidad, aunque no se usará
            u.set_password("demo123")
            db.add(u)
            db.commit()
            print(f"✅ Usuario demo creado: {DEMO_USER_EMAIL}")
        else:
            print(f"✅ Usuario demo ya existe: {DEMO_USER_EMAIL}")


@login_manager.user_loader
def load_user(user_id: str):
    try:
        uid = int(user_id)
    except Exception:
        return None

    with SessionLocal() as db:
        return db.get(User, uid)


# =========================================================
# CHEQUEO DE PAQUETES
# =========================================================
def check_required_packages() -> None:
    """
    Verifica dependencias críticas al iniciar SADI Demo.
    No detiene la app, pero deja mensajes claros en consola.
    """
    import importlib

    required = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("sklearn", "scikit-learn"),
        ("statsmodels.api", "statsmodels"),
        ("factor_analyzer", "factor-analyzer"),
        ("docx", "python-docx"),
    ]

    missing = []

    print("🔎 Verificando dependencias de SADI Demo...")

    for module_name, package_name in required:
        try:
            importlib.import_module(module_name)
            print(f"✅ OK: {package_name}")
        except Exception:
            print(f"❌ FALTA: {package_name}")
            missing.append(package_name)

    if missing:
        print("\n⚠️ Dependencias faltantes detectadas:")
        for pkg in missing:
            print(f"   - {pkg}")

        print("\n👉 Instálalas con:")
        print("pip install " + " ".join(missing))
    else:
        print("✅ Todas las dependencias críticas están disponibles.")


# =========================================================
# ABRIR NAVEGADOR AUTOMÁTICAMENTE
# =========================================================
def open_browser():
    time.sleep(2.5)
    webbrowser.open("http://127.0.0.1:5000")


# =========================================================
# APP FACTORY
# =========================================================
def create_app() -> Flask:
    app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)
    app.config.from_object(Settings)

    # Ajustes demo
    app.config["SECRET_KEY"] = getattr(Settings, "SECRET_KEY", "sadi-demo-secret-key")
    app.config["ES_VERSION_PREMIUM"] = True
    app.config["DEMO_MODE"] = True

    app.jinja_env.auto_reload = True

    login_manager.init_app(app)
    login_manager.login_view = "auth.login"

    # Crear carpetas necesarias
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(MANIFESTS_DIR, exist_ok=True)

    check_required_packages()
    ensure_schema()
    ensure_demo_user()

    with engine.connect() as conn:
        conn.exec_driver_sql("SELECT 1")
    print(f"✅ SQLite DEMO OK -> {DEMO_DB_PATH}")

    @app.before_request
    def _demo_auto_setup():
        # Siempre premium en demo
        session["ES_VERSION_PREMIUM"] = True
        app.config["ES_VERSION_PREMIUM"] = True

        # Autologin del usuario demo
        if not current_user.is_authenticated:
            with SessionLocal() as db:
                demo_user = db.execute(
                    select(User).where(User.email == DEMO_USER_EMAIL)
                ).scalar_one_or_none()
                if demo_user:
                    login_user(demo_user, remember=True)

    @app.get("/")
    def index():
        return redirect(url_for("dataset.dashboard"))

    @app.get("/home", endpoint="home")
    def home():
        return redirect(url_for("dataset.dashboard"))

    # Blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(dataset_bp)
    app.register_blueprint(likert_bp)
    app.register_blueprint(survey_bp)
    app.register_blueprint(multivariate_bp)

    return app
    @app.get("/demo/preparar", endpoint="preparar_demo")
    def preparar_demo():
                import os
                import shutil
                from datetime import datetime
                from sqlalchemy import select
                from flask import flash, redirect, url_for
                from flask_login import current_user
                from models import Dataset

                BASE_DIR = os.path.dirname(os.path.abspath(__file__))

                archivos_demo = [
                    ("Encuesta normal - Demo", "alumnos.csv", "survey_normal", "educacion"),
                    ("Likert 7 puntos - Demo", "dataset_likert_7_sadi.csv", "likert_7", "educacion"),
                    ("Multivariate - Demo", "agronomy_multivariate.csv", "multivariate", "educacion"),
                ]

                with SessionLocal() as db:
                    for titulo, archivo, dataset_type, research_area in archivos_demo:
                        origen = os.path.join(BASE_DIR, "demo_data", archivo)

                        if not os.path.exists(origen):
                            flash(f"No existe el archivo demo: {archivo}", "danger")
                            continue

                        existente = db.execute(
                            select(Dataset).where(
                                Dataset.user_id == current_user.id,
                                Dataset.original_name == archivo
                            )
                        ).scalar_one_or_none()

                        destino = os.path.join(UPLOAD_DIR, archivo)
                        shutil.copyfile(origen, destino)

                        df = pd.read_csv(destino)

                        if existente:
                            existente.title = titulo
                            existente.filename = archivo
                            existente.original_name = archivo
                            existente.delimiter = ","
                            existente.n_rows = int(df.shape[0])
                            existente.n_cols = int(df.shape[1])
                            existente.dataset_type = dataset_type
                            existente.research_area = research_area
                            existente.analysis_cache = None
                            existente.uploaded_at = datetime.utcnow()
                        else:
                            ds = Dataset(
                                user_id=current_user.id,
                                title=titulo,
                                filename=archivo,
                                original_name=archivo,
                                delimiter=",",
                                n_rows=int(df.shape[0]),
                                n_cols=int(df.shape[1]),
                                dataset_type=dataset_type,
                                research_area=research_area,
                                analysis_cache=None,
                                uploaded_at=datetime.utcnow(),
                            )
                            db.add(ds)

                    db.commit()

                flash("Demo preparado: 3 datasets cargados en SADI.", "success")
                return redirect(url_for("dataset.dashboard"))

app = create_app()


if __name__ == "__main__":
    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)