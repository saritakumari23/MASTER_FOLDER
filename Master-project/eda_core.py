from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


@dataclass
class BasicEDAResult:
    summary: Dict[str, Any]
    plot_paths: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _plot_target_distribution(
    df: pd.DataFrame, target_column: Optional[str], out_path: Path
) -> Optional[Path]:
    if not target_column or target_column not in df.columns:
        return None

    s = df[target_column]
    plt.figure(figsize=(6, 4))
    try:
        if pd.api.types.is_numeric_dtype(s):
            sns.histplot(s.dropna(), bins=30, kde=True)
        else:
            vc = s.value_counts().head(30)
            sns.barplot(x=vc.index.astype(str), y=vc.values)
            plt.xticks(rotation=45, ha="right")
        plt.title(f"Target distribution: {target_column}")
        plt.tight_layout()
        _ensure_dir(out_path.parent)
        plt.savefig(out_path, dpi=120)
        return out_path
    finally:
        plt.close()


def _plot_correlation_heatmap(df: pd.DataFrame, out_path: Path) -> Optional[Path]:
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return None

    corr = num_df.corr()
    if corr.empty:
        return None

    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True)
    plt.title("Correlation heatmap")
    plt.tight_layout()
    _ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=120)
    plt.close()
    return out_path


def _plot_missing_bar(df: pd.DataFrame, out_path: Path) -> Optional[Path]:
    missing_pct = df.isna().mean() * 100.0
    if missing_pct.empty:
        return None

    plt.figure(figsize=(max(6, len(df.columns) * 0.25), 4))
    sns.barplot(x=df.columns, y=missing_pct.values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Missing %")
    plt.title("Missing values by column")
    plt.tight_layout()
    _ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=120)
    plt.close()
    return out_path


def run_basic_eda(
    df: pd.DataFrame,
    *,
    target_column: Optional[str],
    out_dir: Path,
) -> BasicEDAResult:
    """
    Lightweight EDA used in DVista Step 3.

    Does not mutate `df`. Returns numeric summaries and saves a few standard plots.
    """

    n_rows, n_cols = df.shape

    # Shape and missing
    missing_pct = df.isna().mean().to_dict()
    missing_pct = {k: float(v) for k, v in missing_pct.items()}

    # Numeric summary
    num_df = df.select_dtypes(include=[np.number])
    numeric_summary: Dict[str, Dict[str, float]] = {}
    if not num_df.empty:
        desc = num_df.describe().T  # rows = columns
        for col, row in desc.iterrows():
            numeric_summary[col] = {
                "mean": float(row.get("mean", np.nan)),
                "std": float(row.get("std", np.nan)),
                "min": float(row.get("min", np.nan)),
                "max": float(row.get("max", np.nan)),
            }

    # Categorical distribution (top categories)
    cat_df = df.select_dtypes(exclude=[np.number])
    categorical_summary: Dict[str, List[Dict[str, Any]]] = {}
    for col in cat_df.columns:
        vc = cat_df[col].value_counts(dropna=False).head(10)
        total = float(len(cat_df[col]))
        categorical_summary[col] = [
            {
                "value": str(idx),
                "count": int(count),
                "proportion": float(count / total) if total > 0 else np.nan,
            }
            for idx, count in vc.items()
        ]

    # Outlier summary via IQR for numeric columns
    outliers: Dict[str, int] = {}
    for col in num_df.columns:
        s = num_df[col]
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            outliers[col] = 0
        else:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers[col] = int(((s < lower) | (s > upper)).sum())

    summary: Dict[str, Any] = {
        "shape": {"n_rows": int(n_rows), "n_cols": int(n_cols)},
        "missing_pct": missing_pct,
        "numeric": numeric_summary,
        "categorical": categorical_summary,
        "outliers_iqr": outliers,
    }

    # Plots
    _ensure_dir(out_dir)
    plot_paths: Dict[str, str] = {}
    target_plot = _plot_target_distribution(
        df, target_column, out_dir / "target_distribution.png"
    )
    if target_plot is not None:
        plot_paths["target_distribution"] = str(target_plot)

    corr_plot = _plot_correlation_heatmap(df, out_dir / "correlation_heatmap.png")
    if corr_plot is not None:
        plot_paths["correlation_heatmap"] = str(corr_plot)

    missing_plot = _plot_missing_bar(df, out_dir / "missing_values.png")
    if missing_plot is not None:
        plot_paths["missing_values"] = str(missing_plot)

    return BasicEDAResult(summary=summary, plot_paths=plot_paths)


