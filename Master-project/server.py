from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import json

from db import DatasetRecord, PreprocessingConfigRecord, PreprocessingProgressRecord, ProjectConfigRecord, get_session, init_db
from eda_core import run_basic_eda
from preprocessing_core import PreprocessingConfig, clean_data, reduce_data, handle_outliers
from preprocessing_workflow import get_steps_for_problem_type, get_step_info, get_next_step, is_step_applicable


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
STATIC_DIR = BASE_DIR / "static"
EDA_DIR = BASE_DIR / "eda_reports"

UPLOAD_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)
EDA_DIR.mkdir(exist_ok=True)


app = FastAPI(title="DVista – Step 1: Dataset Intake")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _guess_dataset_role(df: pd.DataFrame, target_column: str | None) -> str:
    """Heuristic to classify dataset as fact / dimension / transaction."""

    cols = set(df.columns)
    if target_column and target_column in cols:
        return "fact"

    n_rows, n_cols = df.shape
    id_like_cols = [c for c in df.columns if c.lower().endswith("_id") or c.lower() == "id"]
    has_datetime = any(
        pd.api.types.is_datetime64_any_dtype(df[c]) or "date" in c.lower() for c in df.columns
    )

    if n_rows > 5_000 and has_datetime:
        return "transaction"
    if id_like_cols and n_rows < 10_000:
        return "dimension"
    return "dimension"


def _summarize_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    n_rows, n_cols = df.shape
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    missing_pct = {col: float(df[col].isna().mean()) for col in df.columns}
    return {
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "dtypes": dtypes,
        "missing_pct": missing_pct,
    }


@app.on_event("startup")
def on_startup() -> None:
    """Initialize database on startup."""

    init_db()


@app.post("/api/upload-datasets")
async def upload_datasets(
    files: List[UploadFile] = File(...),
    target_column: str | None = Form(None),
) -> Dict[str, Any]:
    """
    STEP 1: Dataset Intake (multi-CSV).

    - Saves each uploaded CSV to disk.
    - Computes basic stats per dataset.
    - Guesses dataset role (fact/dimension/transaction).
    - Stores metadata in SQLite for future use.
    """

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    session = get_session()

    results: List[Dict[str, Any]] = []
    for file in files:
        if not file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only .csv files are supported.")

        dest_path = UPLOAD_DIR / file.filename
        try:
            content = await file.read()
            dest_path.write_bytes(content)
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {exc}")

        try:
            df = pd.read_csv(dest_path)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to read CSV '{file.filename}': {exc}",
            )

        stats = _summarize_dataframe(df)
        role = _guess_dataset_role(df, target_column)
        target_present = "yes" if target_column and target_column in df.columns else "no"

        # Persist in DB
        record = DatasetRecord(
            filename=file.filename,
            stored_path=str(dest_path),
            n_rows=stats["n_rows"],
            n_cols=stats["n_cols"],
            detected_role=role,
            target_present=target_present,
        )
        session.add(record)
        session.commit()
        session.refresh(record)

        results.append(
            {
                "id": record.id,
                "filename": file.filename,
                "stored_path": str(dest_path),
                "role": role,
                "target_present": target_present,
                "summary": stats,
            }
        )

    session.close()

    return {"datasets": results}


@app.get("/api/datasets")
def list_datasets() -> Dict[str, Any]:
    """Return datasets previously uploaded (from DB)."""

    session = get_session()
    records = session.query(DatasetRecord).order_by(DatasetRecord.created_at.desc()).all()
    session.close()

    datasets = [
        {
            "id": r.id,
            "filename": r.filename,
            "stored_path": r.stored_path,
            "n_rows": r.n_rows,
            "n_cols": r.n_cols,
            "role": r.detected_role,
            "target_present": r.target_present,
            "problem_type": r.problem_type,
            "problem_subtype": r.problem_subtype,
            "created_at": r.created_at.isoformat(),
        }
        for r in records
    ]
    return {"datasets": datasets}


@app.post("/api/eda")
def run_eda_for_dataset(
    dataset_id: int = Form(...),
    target_column: str | None = Form(None),
) -> Dict[str, Any]:
    """
    STEP 3: Run basic EDA for a stored dataset.

    Returns JSON summary plus URLs for generated plots.
    """

    session = get_session()
    record = session.query(DatasetRecord).filter(DatasetRecord.id == dataset_id).first()
    session.close()

    if record is None:
        raise HTTPException(status_code=404, detail=f"Dataset with id={dataset_id} not found.")

    csv_path = Path(record.stored_path)
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail=f"CSV not found on disk: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {exc}")

    dataset_dir = EDA_DIR / f"ds_{record.id}"
    eda_result = run_basic_eda(df, target_column=target_column, out_dir=dataset_dir)
    eda_dict = eda_result.to_dict()

    # Convert plot paths to URLs under /eda
    plot_urls: Dict[str, str] = {}
    for key, path in eda_dict.get("plot_paths", {}).items():
        p = Path(path)
        # keep the subdirectory structure under EDA_DIR (e.g. ds_<id>/file.png)
        rel = p.relative_to(EDA_DIR)
        plot_urls[key] = f"/eda/{rel.as_posix()}"

    eda_dict["plot_urls"] = plot_urls

    return eda_dict


@app.post("/api/problem-type")
def save_problem_type(
    problem_type: str = Form(...),
    problem_subtype: str | None = Form(None),
) -> Dict[str, Any]:
    """
    STEP 4: Save problem type identification for the entire project.
    
    This applies to all datasets as future steps work with all datasets together.

    Problem types:
    - classification (with subtypes: binary_classification, multiclass_classification, multilabel_classification)
    - regression
    - clustering
    - dimensionality_reduction
    - anomaly_detection
    - recommendation
    - time_series
    """
    session = get_session()
    
    # Validate problem type
    valid_types = [
        "classification",
        "regression",
        "clustering",
        "dimensionality_reduction",
        "anomaly_detection",
        "recommendation",
        "time_series",
    ]
    if problem_type not in valid_types:
        session.close()
        raise HTTPException(
            status_code=400,
            detail=f"Invalid problem type. Must be one of: {', '.join(valid_types)}",
        )

    # Validate subtype if classification
    if problem_type == "classification":
        if not problem_subtype:
            session.close()
            raise HTTPException(
                status_code=400,
                detail="Classification subtype is required for classification problems.",
            )
        valid_subtypes = ["binary_classification", "multiclass_classification", "multilabel_classification"]
        if problem_subtype not in valid_subtypes:
            session.close()
            raise HTTPException(
                status_code=400,
                detail=f"Invalid problem subtype. Must be one of: {', '.join(valid_subtypes)}",
            )

    # Get or create project config (only one record)
    config = session.query(ProjectConfigRecord).first()
    if config is None:
        config = ProjectConfigRecord(problem_type=problem_type, problem_subtype=problem_subtype)
        session.add(config)
    else:
        config.problem_type = problem_type
        config.problem_subtype = problem_subtype
    
    session.commit()
    session.refresh(config)
    session.close()

    return {
        "problem_type": config.problem_type,
        "problem_subtype": config.problem_subtype,
        "message": "Problem type saved successfully for all datasets",
    }


@app.get("/api/problem-type")
def get_problem_type() -> Dict[str, Any]:
    """Get the current problem type configuration."""
    session = get_session()
    config = session.query(ProjectConfigRecord).first()
    session.close()

    if config is None:
        return {"problem_type": None, "problem_subtype": None}
    
    return {
        "problem_type": config.problem_type,
        "problem_subtype": config.problem_subtype,
    }


@app.get("/api/preprocessing/status")
def get_preprocessing_status() -> Dict[str, Any]:
    """Get current preprocessing status and progress."""
    session = get_session()
    
    # Get problem type
    project_config = session.query(ProjectConfigRecord).first()
    if not project_config or not project_config.problem_type:
        session.close()
        return {
            "status": "not_started",
            "message": "Problem type must be selected first",
            "current_step": None,
            "applicable_steps": [],
        }
    
    # Get preprocessing progress
    progress = session.query(PreprocessingProgressRecord).first()
    session.close()
    
    applicable_steps = get_steps_for_problem_type(project_config.problem_type, project_config.problem_subtype)
    
    if not progress:
        return {
            "status": "not_started",
            "current_step": 1,
            "applicable_steps": applicable_steps,
            "completed_steps": [],
        }
    
    completed_steps = json.loads(progress.completed_steps) if progress.completed_steps else []
    
    return {
        "status": "in_progress" if progress.current_step <= len(applicable_steps) else "completed",
        "current_step": progress.current_step,
        "applicable_steps": applicable_steps,
        "completed_steps": completed_steps,
        "target_column": progress.target_column,
    }


@app.get("/api/preprocessing/steps")
def get_preprocessing_steps() -> Dict[str, Any]:
    """Get all preprocessing steps for current problem type."""
    session = get_session()
    project_config = session.query(ProjectConfigRecord).first()
    session.close()
    
    if not project_config or not project_config.problem_type:
        raise HTTPException(status_code=400, detail="Problem type must be selected first")
    
    steps = get_steps_for_problem_type(project_config.problem_type, project_config.problem_subtype)
    
    return {
        "problem_type": project_config.problem_type,
        "problem_subtype": project_config.problem_subtype,
        "steps": steps,
    }


@app.get("/api/preprocessing/step/{step_number}")
def get_preprocessing_step_info(step_number: int) -> Dict[str, Any]:
    """Get information about a specific preprocessing step."""
    session = get_session()
    project_config = session.query(ProjectConfigRecord).first()
    session.close()
    
    if not project_config or not project_config.problem_type:
        raise HTTPException(status_code=400, detail="Problem type must be selected first")
    
    if not is_step_applicable(step_number, project_config.problem_type):
        raise HTTPException(status_code=400, detail=f"Step {step_number} is not applicable to {project_config.problem_type}")
    
    step_info = get_step_info(step_number)
    if not step_info:
        raise HTTPException(status_code=404, detail=f"Step {step_number} not found")
    
    # Get current progress to see if step is completed
    session = get_session()
    progress = session.query(PreprocessingProgressRecord).first()
    session.close()
    
    completed_steps = json.loads(progress.completed_steps) if progress and progress.completed_steps else []
    
    return {
        **step_info,
        "step_number": step_number,
        "is_completed": step_number in completed_steps,
        "is_current": progress and progress.current_step == step_number,
    }


@app.post("/api/preprocessing/step/1/execute")
def execute_data_cleaning(
    dataset_id: int = Form(...),
    handle_missing: str = Form("mean"),
    drop_duplicates: str = Form("true"),
) -> Dict[str, Any]:
    """
    Execute Step 1: Data Cleaning for a specific dataset.
    
    This step applies to all problem types.
    """
    session = get_session()
    
    # Get dataset
    dataset = session.query(DatasetRecord).filter(DatasetRecord.id == dataset_id).first()
    if not dataset:
        session.close()
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    
    # Get problem type
    project_config = session.query(ProjectConfigRecord).first()
    if not project_config or not project_config.problem_type:
        session.close()
        raise HTTPException(status_code=400, detail="Problem type must be selected first")
    
    session.close()
    
    # Read dataset
    csv_path = Path(dataset.stored_path)
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail=f"CSV file not found: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")
    
    # Execute data cleaning
    config = PreprocessingConfig(
        handle_missing=handle_missing,
        drop_duplicates=(drop_duplicates.lower() == "true"),
    )
    
    initial_shape = df.shape
    df_cleaned = clean_data(df, config)
    final_shape = df_cleaned.shape
    
    # Calculate statistics
    rows_dropped = initial_shape[0] - final_shape[0]
    missing_before = df.isna().sum().to_dict()
    missing_after = df_cleaned.isna().sum().to_dict()
    
    return {
        "dataset_id": dataset_id,
        "filename": dataset.filename,
        "initial_shape": {"rows": int(initial_shape[0]), "cols": int(initial_shape[1])},
        "final_shape": {"rows": int(final_shape[0]), "cols": int(final_shape[1])},
        "rows_dropped": int(rows_dropped),
        "missing_values_before": {k: int(v) for k, v in missing_before.items()},
        "missing_values_after": {k: int(v) for k, v in missing_after.items()},
        "config": {
            "handle_missing": handle_missing,
            "drop_duplicates": drop_duplicates.lower() == "true",
        },
        "message": "Data cleaning completed successfully",
    }


@app.post("/api/preprocessing/step/2/execute")
def execute_data_reduction(
    dataset_id: int = Form(...),
    drop_high_missing: str = Form("0.5"),
    drop_low_variance: str = Form("0.01"),
) -> Dict[str, Any]:
    """
    Execute Step 2: Data Reduction for a specific dataset.
    
    This step applies to all problem types.
    """
    session = get_session()
    
    # Get dataset
    dataset = session.query(DatasetRecord).filter(DatasetRecord.id == dataset_id).first()
    if not dataset:
        session.close()
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    
    # Get problem type
    project_config = session.query(ProjectConfigRecord).first()
    if not project_config or not project_config.problem_type:
        session.close()
        raise HTTPException(status_code=400, detail="Problem type must be selected first")
    
    session.close()
    
    # Read dataset
    csv_path = Path(dataset.stored_path)
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail=f"CSV file not found: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")
    
    # Execute data reduction
    config = PreprocessingConfig(
        drop_high_missing=float(drop_high_missing),
        drop_low_variance=float(drop_low_variance),
    )
    
    initial_shape = df.shape
    initial_columns = list(df.columns)
    df_reduced = reduce_data(df, config)
    final_shape = df_reduced.shape
    final_columns = list(df_reduced.columns)
    
    # Calculate statistics
    columns_dropped = set(initial_columns) - set(final_columns)
    
    return {
        "dataset_id": dataset_id,
        "filename": dataset.filename,
        "initial_shape": {"rows": int(initial_shape[0]), "cols": int(initial_shape[1])},
        "final_shape": {"rows": int(final_shape[0]), "cols": int(final_shape[1])},
        "columns_dropped": list(columns_dropped),
        "columns_dropped_count": len(columns_dropped),
        "config": {
            "drop_high_missing": float(drop_high_missing),
            "drop_low_variance": float(drop_low_variance),
        },
        "message": "Data reduction completed successfully",
    }


@app.post("/api/preprocessing/step/3/execute")
def execute_outlier_handling(
    dataset_id: int = Form(...),
    handle_outliers: str = Form("clip"),
    outlier_method: str = Form("iqr"),
    outlier_threshold: str = Form("3.0"),
) -> Dict[str, Any]:
    """
    Execute Step 3: Outlier Handling for a specific dataset.
    
    This step applies to classification and regression problem types.
    """
    session = get_session()
    
    # Get dataset
    dataset = session.query(DatasetRecord).filter(DatasetRecord.id == dataset_id).first()
    if not dataset:
        session.close()
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    
    # Get problem type
    project_config = session.query(ProjectConfigRecord).first()
    if not project_config or not project_config.problem_type:
        session.close()
        raise HTTPException(status_code=400, detail="Problem type must be selected first")
    
    # Check if step is applicable
    if project_config.problem_type not in ["classification", "regression"]:
        session.close()
        raise HTTPException(
            status_code=400,
            detail=f"Outlier handling is only applicable for classification and regression problems, not {project_config.problem_type}"
        )
    
    session.close()
    
    # Read dataset
    csv_path = Path(dataset.stored_path)
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail=f"CSV file not found: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")
    
    # Execute outlier handling
    config = PreprocessingConfig(
        handle_outliers=handle_outliers,
        outlier_method=outlier_method,
        outlier_threshold=float(outlier_threshold),
    )
    
    initial_shape = df.shape
    initial_rows = initial_shape[0]
    
    # Count outliers before handling
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers_info = {}
    
    if outlier_method == "iqr" and len(numeric_cols) > 0:
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR == 0:
                continue
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outliers_info[col] = len(outliers)
    elif outlier_method == "zscore" and len(numeric_cols) > 0:
        threshold = float(outlier_threshold)
        for col in numeric_cols:
            col_std = df[col].std()
            if col_std == 0:
                continue
            z_scores = np.abs((df[col] - df[col].mean()) / col_std)
            outliers = df[z_scores >= threshold]
            outliers_info[col] = len(outliers)
    
    df_processed = handle_outliers(df, config)
    final_shape = df_processed.shape
    final_rows = final_shape[0]
    
    # Calculate statistics
    rows_removed = initial_rows - final_rows if handle_outliers == "remove" else 0
    
    return {
        "dataset_id": dataset_id,
        "filename": dataset.filename,
        "initial_shape": {"rows": int(initial_shape[0]), "cols": int(initial_shape[1])},
        "final_shape": {"rows": int(final_shape[0]), "cols": int(final_shape[1])},
        "rows_removed": int(rows_removed),
        "outliers_detected": outliers_info,
        "total_outliers_detected": sum(outliers_info.values()),
        "config": {
            "handle_outliers": handle_outliers,
            "outlier_method": outlier_method,
            "outlier_threshold": float(outlier_threshold),
        },
        "message": "Outlier handling completed successfully",
    }


# Preprocessing endpoint temporarily removed (will be replaced with step-by-step endpoints)
# @app.post("/api/preprocessing")
# def run_preprocessing(
#     dataset_id: int = Form(...),
#     target_column: str = Form(...),
#     problem_type: str = Form(...),  # regression / binary_classification / multiclass_classification
#     handle_missing: str = Form("mean"),
#     drop_duplicates: str = Form(None),  # Optional, default to "true"
#     scale_numerical: str = Form("standard"),
#     encode_categorical: str = Form("onehot"),
#     handle_outliers: str = Form("clip"),
#     test_size: str = Form("0.2"),  # Accept as string, convert to float
# ) -> Dict[str, Any]:
#     """
#     STEP 4: Run preprocessing pipeline on a dataset.
#
#     Returns preprocessed train/test splits and summary.
#     """
#     session = get_session()
#     record = session.query(DatasetRecord).filter(DatasetRecord.id == dataset_id).first()
#     session.close()
#
#     if record is None:
#         raise HTTPException(status_code=404, detail=f"Dataset with id={dataset_id} not found.")
#
#     csv_path = Path(record.stored_path)
#     if not csv_path.exists():
#         raise HTTPException(status_code=404, detail=f"CSV not found on disk: {csv_path}")
#
#     try:
#         df = pd.read_csv(csv_path)
#     except Exception as exc:
#         raise HTTPException(status_code=400, detail=f"Failed to read CSV: {exc}")
#
#     if target_column not in df.columns:
#         raise HTTPException(
#             status_code=400, detail=f"Target column '{target_column}' not found in dataset."
#         )
#
#     # Create preprocessing config
#     config = PreprocessingConfig(
#         handle_missing=handle_missing,
#         drop_duplicates=(drop_duplicates or "true").lower() == "true",
#         scale_numerical=scale_numerical,
#         encode_categorical=encode_categorical,
#         handle_outliers=handle_outliers,
#         test_size=float(test_size),
#     )
#
#     # Run preprocessing pipeline
#     try:
#         result = run_preprocessing_pipeline(df, target_column, config, problem_type)
#     except Exception as exc:
#         import traceback
#         error_detail = f"Preprocessing failed: {str(exc)}\n{traceback.format_exc()}"
#         raise HTTPException(status_code=400, detail=error_detail)
#
#     # Save config to DB
#     session = get_session()
#     config_record = PreprocessingConfigRecord(
#         dataset_id=dataset_id,
#         target_column=target_column,
#         problem_type=problem_type,
#         config_json=json.dumps(config.__dict__),
#     )
#     session.add(config_record)
#     session.commit()
#     session.close()
#
#     return result.to_dict()


# Serve generated EDA plots (mount this before root static so /eda is not shadowed)
app.mount("/eda", StaticFiles(directory=EDA_DIR), name="eda")
# Serve static frontend (index.html, JS, CSS)
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


def get_app() -> FastAPI:
    """Helper for ASGI servers."""

    return app

