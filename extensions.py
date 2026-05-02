from __future__ import annotations

import os

from flask_login import LoginManager
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from config import DATABASE_URL

login_manager = LoginManager()
login_manager.login_view = "auth.login"


def _resolve_database_url() -> str:
    """
    Prioridad:
    1) Variable de entorno DATABASE_URL (útil para demo o despliegue)
    2) DATABASE_URL definido en config.py
    """
    env_url = os.getenv("DATABASE_URL")
    if env_url and env_url.strip():
        return env_url.strip()
    return DATABASE_URL


def _is_sqlite_url(db_url: str) -> bool:
    return db_url.startswith("sqlite")


def _build_engine(db_url: str):
    """
    Crea el engine con opciones compatibles según el motor.
    """
    common_kwargs = {
        "future": True,
    }

    if _is_sqlite_url(db_url):
        return create_engine(
            db_url,
            connect_args={"check_same_thread": False},
            **common_kwargs,
        )

    return create_engine(
        db_url,
        pool_pre_ping=True,
        **common_kwargs,
    )


DATABASE_URL_RESOLVED = _resolve_database_url()

engine = _build_engine(DATABASE_URL_RESOLVED)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


def reconfigure_database(new_database_url: str) -> None:
    """
    Reconfigura engine y SessionLocal en runtime.
    Muy útil para demo_sadi.py cuando quiera forzar SQLite.
    """
    global engine, SessionLocal, DATABASE_URL_RESOLVED

    DATABASE_URL_RESOLVED = new_database_url
    engine = _build_engine(new_database_url)

    SessionLocal.configure(bind=engine)

    print("✅ Base de datos reconfigurada a:", engine.url)


def debug_database_connection() -> None:
    """
    Muestra información de conexión sin romper si el motor es SQLite.
    """
    print("ENGINE URL:", engine.url)

    try:
        with SessionLocal() as db:
            if _is_sqlite_url(str(engine.url)):
                row = db.execute(text("SELECT sqlite_version()")).fetchone()
                print("SQLITE VERSION:", row[0] if row else "desconocida")
            else:
                row = db.execute(
                #    text("SELECT current_database(), current_schema()")
                ).fetchone()
                print("DB ACTUAL:", row)
    except Exception as e:
        print("⚠️ No se pudo verificar la conexión a la base de datos:", e)