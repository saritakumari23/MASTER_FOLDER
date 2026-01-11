from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import Column, DateTime, Integer, String, create_engine, text
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
    problem_type: str = Column(String, nullable=True)  # classification / regression / clustering / dimensionality_reduction / anomaly_detection / recommendation / time_series
    problem_subtype: str = Column(String, nullable=True)  # binary_classification / multiclass_classification / multilabel_classification
    created_at: datetime = Column(DateTime, default=datetime.utcnow, nullable=False)


class ProjectConfigRecord(Base):
    __tablename__ = "project_config"

    id: int = Column(Integer, primary_key=True, index=True)
    problem_type: str = Column(String, nullable=True)  # classification / regression / clustering / etc.
    problem_subtype: str = Column(String, nullable=True)  # binary_classification / multiclass_classification / multilabel_classification
    updated_at: datetime = Column(DateTime, default=datetime.utcnow, nullable=False)


class PreprocessingProgressRecord(Base):
    __tablename__ = "preprocessing_progress"

    id: int = Column(Integer, primary_key=True, index=True)
    current_step: int = Column(Integer, nullable=False, default=1)  # 1-7, current step number
    target_column: str = Column(String, nullable=True)  # Target column name (if applicable)
    config_json: str = Column(String, nullable=True)  # JSON string of current PreprocessingConfig
    completed_steps: str = Column(String, nullable=True)  # JSON array of completed step numbers
    created_at: datetime = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: datetime = Column(DateTime, default=datetime.utcnow, nullable=False)


class PreprocessingConfigRecord(Base):
    __tablename__ = "preprocessing_configs"

    id: int = Column(Integer, primary_key=True, index=True)
    dataset_id: int = Column(Integer, nullable=False, index=True)
    target_column: str = Column(String, nullable=False)
    problem_type: str = Column(String, nullable=False)  # regression / binary_classification / multiclass_classification
    config_json: str = Column(String, nullable=False)  # JSON string of PreprocessingConfig
    created_at: datetime = Column(DateTime, default=datetime.utcnow, nullable=False)


def init_db() -> None:
    """Create tables if they do not exist and migrate existing tables."""
    
    # Create all tables (for new databases)
    Base.metadata.create_all(bind=engine)
    
    # Migrate existing tables: Add new columns if they don't exist
    _migrate_add_problem_type_columns()


def _migrate_add_problem_type_columns() -> None:
    """Add problem_type and problem_subtype columns to datasets table if they don't exist."""
    try:
        with engine.begin() as conn:  # Use begin() for automatic transaction management
            # Check if columns exist by querying table info
            result = conn.execute(text("PRAGMA table_info(datasets)"))
            columns = [row[1] for row in result.fetchall()]
            
            # Add problem_type column if it doesn't exist
            if "problem_type" not in columns:
                conn.execute(text("ALTER TABLE datasets ADD COLUMN problem_type VARCHAR"))
            
            # Add problem_subtype column if it doesn't exist
            if "problem_subtype" not in columns:
                conn.execute(text("ALTER TABLE datasets ADD COLUMN problem_subtype VARCHAR"))
    except Exception as e:
        # If migration fails, log but don't crash (columns might already exist or other issue)
        # This can happen if the table doesn't exist yet (will be created by create_all above)
        pass  # Silent fail is OK here - columns will be created by create_all for new databases


def get_session() -> Session:
    """Return a new SQLAlchemy session."""

    return SessionLocal()


