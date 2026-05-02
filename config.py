from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# =========================
# 📁 DIRECTORIOS SADI
# =========================
STATIC_DIR = str(BASE_DIR / "static")
TEMPLATES_DIR = str(BASE_DIR / "templates")
PLOTS_DIR = str(BASE_DIR / "static" / "plots")
UPLOAD_DIR = str(BASE_DIR / "uploads")
MANIFESTS_DIR = str(BASE_DIR / "manifests")   # 👈 NUEVO

# =========================
# 🗄️ DATABASE
# =========================
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://postgres@127.0.0.1:5432/sadi",
)

# =========================
# ⚙️ APP
# =========================
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")
FLASK_ENV = os.getenv("FLASK_ENV", "development")


class Settings:
    SECRET_KEY = SECRET_KEY
    ENV = FLASK_ENV
    TEMPLATES_AUTO_RELOAD = True