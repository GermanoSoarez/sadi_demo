from __future__ import annotations

from datetime import datetime
from typing import Optional

from flask_login import UserMixin
from sqlalchemy import Boolean, DateTime, ForeignKey, String, JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from werkzeug.security import check_password_hash, generate_password_hash


class Base(DeclarativeBase):
    pass


class User(UserMixin, Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)

    is_premium: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    premium_until: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    datasets: Mapped[list["Dataset"]] = relationship(
        "Dataset",
        back_populates="owner",
        cascade="all, delete-orphan",
    )

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))

    title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    filename: Mapped[str] = mapped_column(String(255))
    original_name: Mapped[str] = mapped_column(String(255))

    delimiter: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)
    n_rows: Mapped[int] = mapped_column(default=0)
    n_cols: Mapped[int] = mapped_column(default=0)

    dataset_type: Mapped[str] = mapped_column(String(32), default="no_definido", nullable=False)
    research_area: Mapped[str] = mapped_column(String(32), default="general", nullable=False)
    analysis_cache: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    uploaded_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    owner: Mapped[User] = relationship(back_populates="datasets")