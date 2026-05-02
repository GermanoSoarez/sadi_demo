from __future__ import annotations

from flask import Blueprint, flash, redirect, render_template, request, session, url_for
from flask_login import current_user, login_required, login_user, logout_user

from extensions import SessionLocal
from models import User

auth_bp = Blueprint("auth", __name__)


@auth_bp.get("/login")
def login():
    return render_template("login.html")


@auth_bp.post("/login")
def login_post():
    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""

    with SessionLocal() as db:
        u = db.query(User).filter(User.email == email).first()
        if not u or not u.check_password(password):
            flash("Credenciales inválidas.", "danger")
            return redirect(url_for("auth.login"))
        login_user(u)

    return redirect(url_for("dataset.dashboard"))


@auth_bp.get("/logout")
@login_required
def logout():
    logout_user()
    flash("Sesión cerrada correctamente.", "info")
    return redirect(url_for("auth.login"))


@auth_bp.get("/register")
def register():
    return render_template("register.html")


@auth_bp.post("/register")
def register_post():
    name = (request.form.get("name") or "").strip()
    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""

    if not email or not password:
        flash("Debes completar correo y contraseña.", "warning")
        return redirect(url_for("auth.register"))

    with SessionLocal() as db:
        exists = db.query(User).filter(User.email == email).first()
        if exists:
            flash("Ya existe un usuario con ese correo.", "warning")
            return redirect(url_for("auth.register"))

        u = User(name=name or None, email=email)
        u.set_password(password)
        db.add(u)
        db.commit()

    flash("Usuario registrado correctamente. Ya puedes iniciar sesión.", "success")
    return redirect(url_for("auth.login"))


@auth_bp.get("/configurar_modo_sadi")
@login_required
def configurar_modo_sadi():
    is_premium = bool(session.get("ES_VERSION_PREMIUM", False))
    return render_template("premium_toggle.html", is_premium=is_premium)


@auth_bp.post("/configurar_modo_sadi")
@login_required
def configurar_modo_sadi_post():
    is_premium = request.form.get("is_premium") == "1"

    session["ES_VERSION_PREMIUM"] = is_premium
    session.modified = True

    if is_premium:
        flash("Modo SADI 2.0 Premium activado.", "success")
    else:
        flash("Modo SADI básico activado.", "info")

    return redirect(url_for("dataset.dashboard"))