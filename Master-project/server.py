from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import json

from db import DatasetRecord, PreprocessingConfigRecord, get_session, init_db
from eda_core import run_basic_eda
from preprocessing_core import PreprocessingConfig, run_preprocessing_pipeline


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


@app.post("/api/preprocessing")
def run_preprocessing(
    dataset_id: int = Form(...),
    target_column: str = Form(...),
    problem_type: str = Form(...),  # regression / binary_classification / multiclass_classification
    handle_missing: str = Form("mean"),
    drop_duplicates: str = Form(None),  # Optional, default to "true"
    scale_numerical: str = Form("standard"),
    encode_categorical: str = Form("onehot"),
    handle_outliers: str = Form("clip"),
    test_size: str = Form("0.2"),  # Accept as string, convert to float
) -> Dict[str, Any]:
    """
    STEP 4: Run preprocessing pipeline on a dataset.

    Returns preprocessed train/test splits and summary.
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

    if target_column not in df.columns:
        raise HTTPException(
            status_code=400, detail=f"Target column '{target_column}' not found in dataset."
        )

    # Create preprocessing config
    config = PreprocessingConfig(
        handle_missing=handle_missing,
        drop_duplicates=(drop_duplicates or "true").lower() == "true",
        scale_numerical=scale_numerical,
        encode_categorical=encode_categorical,
        handle_outliers=handle_outliers,
        test_size=float(test_size),
    )

    # Run preprocessing pipeline
    try:
        result = run_preprocessing_pipeline(df, target_column, config, problem_type)
    except Exception as exc:
        import traceback
        error_detail = f"Preprocessing failed: {str(exc)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=400, detail=error_detail)

    # Save config to DB
    session = get_session()
    config_record = PreprocessingConfigRecord(
        dataset_id=dataset_id,
        target_column=target_column,
        problem_type=problem_type,
        config_json=json.dumps(config.__dict__),
    )
    session.add(config_record)
    session.commit()
    session.close()

    return result.to_dict()


# Serve generated EDA plots (mount this before root static so /eda is not shadowed)
app.mount("/eda", StaticFiles(directory=EDA_DIR), name="eda")
# Serve static frontend (index.html, JS, CSS)
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


def get_app() -> FastAPI:
    """Helper for ASGI servers."""

    return app

