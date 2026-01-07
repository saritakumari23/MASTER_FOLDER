from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import Column, DateTime, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "dvista.db"
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"


class Base(DeclarativeBase):
    pass


engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class DatasetRecord(Base):
    __tablename__ = "datasets"

    id: int = Column(Integer, primary_key=True, index=True)
    filename: str = Column(String, index=True, nullable=False)
    stored_path: str = Column(String, nullable=False)
    n_rows: int = Column(Integer, nullable=False)
    n_cols: int = Column(Integer, nullable=False)
    detected_role: str = Column(String, nullable=False)  # fact / dimension / transaction
    target_present: str = Column(String, nullable=True)  # "yes" / "no" / None
    created_at: datetime = Column(DateTime, default=datetime.utcnow, nullable=False)


class PreprocessingConfigRecord(Base):
    __tablename__ = "preprocessing_configs"

    id: int = Column(Integer, primary_key=True, index=True)
    dataset_id: int = Column(Integer, nullable=False, index=True)
    target_column: str = Column(String, nullable=False)
    problem_type: str = Column(String, nullable=False)  # regression / binary_classification / multiclass_classification
    config_json: str = Column(String, nullable=False)  # JSON string of PreprocessingConfig
    created_at: datetime = Column(DateTime, default=datetime.utcnow, nullable=False)


def init_db() -> None:
    """Create tables if they do not exist."""

    Base.metadata.create_all(bind=engine)


def get_session() -> Session:
    """Return a new SQLAlchemy session."""

    return SessionLocal()


