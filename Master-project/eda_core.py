from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


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


def _plot_correlation_heatmap(df: pd.DataFrame, out_path: Path) -> Optional[Path]:
    """Basic correlation heatmap (fallback)."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return None

    corr = num_df.corr()
    if corr.empty:
        return None

    plt.figure(figsize=(max(8, num_df.shape[1] * 0.5), max(6, num_df.shape[1] * 0.5)))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True, annot=False)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    _ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=120)
    plt.close()
    return out_path


def _plot_correlation_heatmap_advanced(
    df: pd.DataFrame, target_column: Optional[str], out_path: Path
) -> Optional[Path]:
    """Enhanced correlation heatmap with annotations and target correlation."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return None

    corr = num_df.corr()
    if corr.empty:
        return None

    # Calculate figure size based on number of columns
    n_cols = corr.shape[0]
    fig_size = max(8, min(20, n_cols * 0.5))
    
    plt.figure(figsize=(fig_size, fig_size))
    mask = np.triu(np.ones_like(corr, dtype=bool))  # Mask upper triangle
    sns.heatmap(
        corr,
        mask=mask,
        cmap="coolwarm",
        center=0,
        square=True,
        annot=True,
        fmt=".2f",
        cbar_kws={"shrink": 0.8},
        linewidths=0.5,
    )
    plt.title("Correlation Heatmap (Lower Triangle)")
    plt.tight_layout()
    _ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return out_path


def _plot_target_correlations(
    df: pd.DataFrame, target_column: Optional[str], out_path: Path
) -> Optional[Path]:
    """Plot correlations of all features with target variable."""
    if not target_column or target_column not in df.columns:
        return None

    num_df = df.select_dtypes(include=[np.number])
    if target_column not in num_df.columns or num_df.shape[1] < 2:
        return None

    # Calculate correlations with target
    target_corr = num_df.corr()[target_column].drop(target_column).sort_values(
        key=abs, ascending=False
    )

    if target_corr.empty:
        return None

    plt.figure(figsize=(max(8, len(target_corr) * 0.4), 6))
    colors = ["#e74c3c" if x < 0 else "#2ecc71" for x in target_corr.values]
    bars = plt.barh(range(len(target_corr)), target_corr.values, color=colors)
    plt.yticks(range(len(target_corr)), target_corr.index)
    plt.xlabel("Correlation with Target")
    plt.title(f"Feature Correlations with Target: {target_column}")
    plt.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
    plt.grid(axis="x", alpha=0.3)
    
    # Add value labels on bars
    for i, (idx, val) in enumerate(target_corr.items()):
        plt.text(val, i, f" {val:.3f}", va="center", fontsize=9)
    
    plt.tight_layout()
    _ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return out_path


def _plot_boxplots_outliers(df: pd.DataFrame, out_path: Path, max_cols: int = 10) -> Optional[Path]:
    """Plot box plots for numeric columns to visualize outliers."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        return None

    # Limit number of columns to avoid too many plots
    cols_to_plot = num_df.columns[:max_cols].tolist()
    n_cols = len(cols_to_plot)
    
    if n_cols == 0:
        return None

    # Create subplots grid
    n_rows = (n_cols + 2) // 3  # 3 columns per row
    fig, axes_array = plt.subplots(n_rows, 3, figsize=(15, n_rows * 4))
    
    # Properly flatten axes array to handle all cases
    # When n_rows == 1 and n_cols == 1: axes_array is a single Axes object
    # When n_rows == 1 and n_cols > 1: axes_array is a 1D numpy array
    # When n_rows > 1: axes_array is a 2D numpy array
    if hasattr(axes_array, 'flatten'):
        # It's a numpy array (single or multi-dimensional)
        axes = axes_array.flatten()
    elif hasattr(axes_array, 'set_xlabel'):
        # It's a single Axes object
        axes = [axes_array]
    else:
        # Try to convert to array and flatten
        try:
            axes = np.array(axes_array).flatten()
        except (TypeError, ValueError):
            axes = [axes_array] if axes_array else []
    
    # Ensure axes is always a list for easy indexing
    if not isinstance(axes, list):
        axes = list(axes) if hasattr(axes, '__iter__') and not isinstance(axes, str) else [axes]
    
    # Calculate outliers for title
    for idx, col in enumerate(cols_to_plot):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        data = num_df[col].dropna()
        if len(data) > 0:
            # Calculate outliers using IQR
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                outlier_count = 0
            else:
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outlier_count = int(((num_df[col] < lower) | (num_df[col] > upper)).sum())
            
            sns.boxplot(y=data, ax=ax)
            ax.set_title(f"{col}\n(Outliers: {outlier_count})", fontsize=9)
            ax.set_ylabel("")
    
    # Hide extra subplots
    for idx in range(n_cols, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Box Plots for Outlier Detection (Top 10 Numeric Columns)", fontsize=14, y=1.02)
    plt.tight_layout()
    _ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return out_path


def _plot_distributions(df: pd.DataFrame, out_path: Path, max_cols: int = 12) -> Optional[Path]:
    """Plot distribution (histogram + KDE) for all numeric columns."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        return None

    cols_to_plot = num_df.columns[:max_cols].tolist()
    n_cols = len(cols_to_plot)
    
    if n_cols == 0:
        return None

    # Create subplots grid
    n_rows = (n_cols + 3) // 4  # 4 columns per row
    fig, axes_array = plt.subplots(n_rows, 4, figsize=(20, n_rows * 4))
    
    # Properly flatten axes array to handle all cases
    # When n_rows == 1 and n_cols == 1: axes_array is a single Axes object
    # When n_rows == 1 and n_cols > 1: axes_array is a 1D numpy array
    # When n_rows > 1: axes_array is a 2D numpy array
    if hasattr(axes_array, 'flatten'):
        # It's a numpy array (single or multi-dimensional)
        axes = axes_array.flatten()
    elif hasattr(axes_array, 'set_xlabel'):
        # It's a single Axes object
        axes = [axes_array]
    else:
        # Try to convert to array and flatten
        try:
            axes = np.array(axes_array).flatten()
        except (TypeError, ValueError):
            axes = [axes_array] if axes_array else []
    
    # Ensure axes is always a list for easy indexing
    if not isinstance(axes, list):
        axes = list(axes) if hasattr(axes, '__iter__') and not isinstance(axes, str) else [axes]

    for idx, col in enumerate(cols_to_plot):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        data = num_df[col].dropna()
        if len(data) > 0:
            sns.histplot(data, kde=True, ax=ax, bins=30)
            ax.set_title(f"{col}\n(Mean: {data.mean():.2f}, Std: {data.std():.2f})", fontsize=9)
            ax.set_ylabel("")
            ax.set_xlabel("")
    
    # Hide extra subplots
    for idx in range(n_cols, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Distribution Plots for Numeric Features (Top 12)", fontsize=14, y=1.02)
    plt.tight_layout()
    _ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return out_path


def _plot_pairplot_top_features(
    df: pd.DataFrame, target_column: Optional[str], out_path: Path, max_features: int = 6
) -> Optional[Path]:
    """Create pair plot for top correlated features (including target if specified)."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return None

    # Select features to plot
    features_to_plot = []
    
    if target_column and target_column in num_df.columns:
        # Get top features correlated with target
        available_features = [c for c in num_df.columns if c != target_column]
        if len(available_features) > 0:
            target_corr = num_df[available_features + [target_column]].corr()[target_column].abs().drop(target_column).sort_values(ascending=False)
            top_features = target_corr.head(max_features - 1).index.tolist()
            features_to_plot = top_features + [target_column]
        else:
            return None
    else:
        # Get top features by variance or select first few
        features_to_plot = num_df.columns[:max_features].tolist()

    if len(features_to_plot) < 2:
        return None

    # Limit to avoid memory issues
    sample_size = min(1000, len(num_df))
    sample_df = num_df[features_to_plot].sample(n=sample_size, random_state=42) if len(num_df) > sample_size else num_df[features_to_plot]

    try:
        g = sns.PairGrid(sample_df, diag_sharey=False)
        g.map_upper(sns.scatterplot, alpha=0.3, s=10)
        g.map_lower(sns.scatterplot, alpha=0.3, s=10)
        g.map_diag(sns.histplot, kde=True, bins=20)
        plt.suptitle("Pair Plot: Feature Relationships", fontsize=12, y=1.02)
        plt.tight_layout()
        _ensure_dir(out_path.parent)
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
        return out_path
    except Exception as e:
        # If pair plot fails (memory/time), return None
        plt.close()
        return None


def run_basic_eda(
    df: pd.DataFrame,
    *,
    target_column: Optional[str],
    out_dir: Path,
) -> BasicEDAResult:
    """
    Advanced EDA used in DVista Step 3.

    Performs comprehensive exploratory data analysis including:
    - Basic statistics and missing value analysis
    - Advanced statistics (skewness, kurtosis)
    - Correlation analysis (full matrix, target correlations, top correlations)
    - Distribution analysis (histograms, box plots)
    - Outlier detection
    - Pair plots for feature relationships

    Does not mutate `df`. Returns comprehensive summaries and saves multiple plots.
    """

    n_rows, n_cols = df.shape

    # Shape and missing
    missing_pct = df.isna().mean().to_dict()
    missing_pct = {k: float(v) for k, v in missing_pct.items()}

    # Numeric summary with advanced statistics
    num_df = df.select_dtypes(include=[np.number])
    numeric_summary: Dict[str, Dict[str, float]] = {}
    numeric_advanced: Dict[str, Dict[str, float]] = {}
    
    if not num_df.empty:
        desc = num_df.describe().T  # rows = columns
        for col, row in desc.iterrows():
            s = num_df[col].dropna()
            if len(s) > 0:
                numeric_summary[col] = {
                    "mean": float(row.get("mean", np.nan)),
                    "std": float(row.get("std", np.nan)),
                    "min": float(row.get("min", np.nan)),
                    "max": float(row.get("max", np.nan)),
                    "median": float(row.get("50%", np.nan)),
                    "q25": float(row.get("25%", np.nan)),
                    "q75": float(row.get("75%", np.nan)),
                }
                
                # Advanced statistics
                try:
                    skew_val = float(stats.skew(s))
                    kurt_val = float(stats.kurtosis(s))
                    cv = float(s.std() / s.mean()) if s.mean() != 0 else np.nan  # Coefficient of variation
                except Exception:
                    skew_val = np.nan
                    kurt_val = np.nan
                    cv = np.nan
                
                numeric_advanced[col] = {
                    "skewness": skew_val,
                    "kurtosis": kurt_val,
                    "coefficient_of_variation": cv,
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

    # Outlier summary via IQR and Z-score for numeric columns
    outliers_iqr: Dict[str, int] = {}
    outliers_zscore: Dict[str, int] = {}
    
    for col in num_df.columns:
        s = num_df[col].dropna()
        if len(s) == 0:
            outliers_iqr[col] = 0
            outliers_zscore[col] = 0
            continue
            
        # IQR method
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            outliers_iqr[col] = 0
        else:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers_iqr[col] = int(((num_df[col] < lower) | (num_df[col] > upper)).sum())
        
        # Z-score method
        mean_val = s.mean()
        std_val = s.std()
        if pd.isna(std_val) or std_val == 0:
            outliers_zscore[col] = 0
        else:
            z_scores = np.abs((num_df[col] - mean_val) / std_val)
            outliers_zscore[col] = int((z_scores > 3).sum())

    # Correlation analysis
    correlation_analysis: Dict[str, Any] = {}
    if not num_df.empty and num_df.shape[1] >= 2:
        corr_matrix = num_df.corr()
        
        # Overall correlation summary
        corr_values = corr_matrix.values
        np.fill_diagonal(corr_values, np.nan)  # Exclude diagonal
        corr_flat = corr_values.flatten()
        corr_flat = corr_flat[~np.isnan(corr_flat)]
        
        correlation_analysis["overall"] = {
            "mean_absolute_correlation": float(np.abs(corr_flat).mean()),
            "max_correlation": float(np.abs(corr_flat).max()),
            "min_correlation": float(corr_flat.min()),
        }
        
        # Top correlations (excluding target if specified)
        if target_column and target_column in num_df.columns:
            target_corr = corr_matrix[target_column].drop(target_column).sort_values(
                key=abs, ascending=False
            )
            correlation_analysis["target_correlations"] = {
                col: float(val)
                for col, val in target_corr.head(10).items()
            }
        
        # Find strongest feature pairs (excluding target)
        corr_no_target = corr_matrix.copy()
        if target_column and target_column in corr_no_target.columns:
            corr_no_target = corr_no_target.drop(columns=[target_column], index=[target_column])
        
        if not corr_no_target.empty:
            # Get upper triangle pairs
            mask = np.triu(np.ones_like(corr_no_target, dtype=bool), k=1)
            corr_pairs = corr_no_target.where(mask).stack().reset_index()
            corr_pairs.columns = ["feature1", "feature2", "correlation"]
            corr_pairs["abs_correlation"] = corr_pairs["correlation"].abs()
            top_pairs = corr_pairs.nlargest(10, "abs_correlation")
            
            correlation_analysis["top_feature_pairs"] = [
                {
                    "feature1": str(row["feature1"]),
                    "feature2": str(row["feature2"]),
                    "correlation": float(row["correlation"]),
                }
                for _, row in top_pairs.iterrows()
            ]

    # Build comprehensive summary
    summary: Dict[str, Any] = {
        "shape": {"n_rows": int(n_rows), "n_cols": int(n_cols)},
        "missing_pct": missing_pct,
        "numeric": numeric_summary,
        "numeric_advanced": numeric_advanced,
        "categorical": categorical_summary,
        "outliers_iqr": outliers_iqr,
        "outliers_zscore": outliers_zscore,
        "correlation_analysis": correlation_analysis,
    }

    # Generate all plots
    _ensure_dir(out_dir)
    plot_paths: Dict[str, str] = {}
    
    # Basic plots
    target_plot = _plot_target_distribution(
        df, target_column, out_dir / "target_distribution.png"
    )
    if target_plot is not None:
        plot_paths["target_distribution"] = str(target_plot)

    # Enhanced correlation heatmap (replaces basic one)
    corr_plot_advanced = _plot_correlation_heatmap_advanced(
        df, target_column, out_dir / "correlation_heatmap_advanced.png"
    )
    if corr_plot_advanced is not None:
        plot_paths["correlation_heatmap_advanced"] = str(corr_plot_advanced)
    else:
        # Fallback to basic if advanced fails
        corr_plot = _plot_correlation_heatmap(df, out_dir / "correlation_heatmap.png")
        if corr_plot is not None:
            plot_paths["correlation_heatmap"] = str(corr_plot)

    # Target correlations plot
    target_corr_plot = _plot_target_correlations(
        df, target_column, out_dir / "target_correlations.png"
    )
    if target_corr_plot is not None:
        plot_paths["target_correlations"] = str(target_corr_plot)

    missing_plot = _plot_missing_bar(df, out_dir / "missing_values.png")
    if missing_plot is not None:
        plot_paths["missing_values"] = str(missing_plot)

    # Advanced plots
    boxplot_plot = _plot_boxplots_outliers(df, out_dir / "boxplots_outliers.png")
    if boxplot_plot is not None:
        plot_paths["boxplots_outliers"] = str(boxplot_plot)

    dist_plot = _plot_distributions(df, out_dir / "distributions.png")
    if dist_plot is not None:
        plot_paths["distributions"] = str(dist_plot)

    pairplot_plot = _plot_pairplot_top_features(
        df, target_column, out_dir / "pairplot_top_features.png"
    )
    if pairplot_plot is not None:
        plot_paths["pairplot_top_features"] = str(pairplot_plot)

    return BasicEDAResult(summary=summary, plot_paths=plot_paths)


