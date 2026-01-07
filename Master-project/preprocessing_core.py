"""
DVista Preprocessing Pipeline

Handles all data preprocessing steps:
- Data Cleaning
- Data Integration
- Data Transformation
- Data Reduction
- Feature Engineering & Selection
- Scaling / Normalization
- Categorical Encoding
- Outlier Handling
- Train-test split / CV
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""

    # Data Cleaning
    handle_missing: str = "mean"  # mean / median / mode / drop / forward_fill
    drop_duplicates: bool = True

    # Data Integration
    join_datasets: List[Dict[str, Any]] = None  # List of join configs

    # Data Transformation
    log_transform: List[str] = None  # Columns to log-transform
    power_transform: List[str] = None  # Columns to power-transform

    # Data Reduction
    drop_high_missing: float = 0.5  # Drop columns with >50% missing
    drop_low_variance: float = 0.01  # Drop columns with variance < threshold

    # Feature Engineering
    create_interactions: bool = False
    create_polynomials: bool = False
    polynomial_degree: int = 2

    # Scaling / Normalization
    scale_numerical: str = "standard"  # standard / minmax / robust / none
    encode_categorical: str = "onehot"  # onehot / label / target / none

    # Outlier Handling
    handle_outliers: str = "clip"  # clip / remove / none
    outlier_method: str = "iqr"  # iqr / zscore
    outlier_threshold: float = 3.0  # For zscore

    # Train-test split
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True  # For classification

    def __post_init__(self):
        if self.join_datasets is None:
            self.join_datasets = []
        if self.log_transform is None:
            self.log_transform = []
        if self.power_transform is None:
            self.power_transform = []


@dataclass
class PreprocessingResult:
    """Result of preprocessing pipeline."""

    train_df: pd.DataFrame
    test_df: pd.DataFrame
    feature_names: List[str]
    target_name: str
    n_features: int
    n_train: int
    n_test: int
    preprocessing_summary: Dict[str, Any]
    transformer: Optional[ColumnTransformer] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "n_features": self.n_features,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "feature_names": self.feature_names,
            "target_name": self.target_name,
            "preprocessing_summary": self.preprocessing_summary,
            "train_preview": self.train_df.head(10).to_dict(orient="records"),
            "test_preview": self.test_df.head(10).to_dict(orient="records"),
        }


def clean_data(df: pd.DataFrame, config: PreprocessingConfig) -> pd.DataFrame:
    """Step 1: Data Cleaning."""
    df = df.copy()

    # Drop duplicates
    if config.drop_duplicates:
        n_before = len(df)
        df = df.drop_duplicates()
        n_dropped = n_before - len(df)

    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    if config.handle_missing == "mean" and len(numeric_cols) > 0:
        imputer = SimpleImputer(strategy="mean")
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    elif config.handle_missing == "median" and len(numeric_cols) > 0:
        imputer = SimpleImputer(strategy="median")
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    elif config.handle_missing == "mode" and len(categorical_cols) > 0:
        imputer = SimpleImputer(strategy="most_frequent")
        df[categorical_cols] = imputer.fit_transform(df[categorical_cols])
    elif config.handle_missing == "forward_fill":
        df = df.ffill().bfill()
    elif config.handle_missing == "drop":
        df = df.dropna()

    return df


def reduce_data(df: pd.DataFrame, config: PreprocessingConfig) -> pd.DataFrame:
    """Step 2: Data Reduction."""
    df = df.copy()

    # Drop high missing columns
    missing_pct = df.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > config.drop_high_missing].index.tolist()
    df = df.drop(columns=cols_to_drop)

    # Drop low variance columns (numeric only)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        variances = df[numeric_cols].var()
        low_var_cols = variances[variances < config.drop_low_variance].index.tolist()
        df = df.drop(columns=low_var_cols)

    return df


def transform_features(df: pd.DataFrame, config: PreprocessingConfig) -> pd.DataFrame:
    """Step 3: Data Transformation."""
    df = df.copy()

    # Log transform
    for col in config.log_transform:
        if col in df.columns:
            df[f"{col}_log"] = np.log1p(df[col])

    # Power transform
    for col in config.power_transform:
        if col in df.columns:
            df[f"{col}_power"] = np.power(df[col], 2)

    return df


def handle_outliers(df: pd.DataFrame, config: PreprocessingConfig) -> pd.DataFrame:
    """Step 4: Outlier Handling."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if config.handle_outliers == "none" or len(numeric_cols) == 0:
        return df

    for col in numeric_cols:
        if config.outlier_method == "iqr":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR == 0:  # Skip if no variance
                continue
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            if config.handle_outliers == "clip":
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            elif config.handle_outliers == "remove":
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

        elif config.outlier_method == "zscore":
            col_std = df[col].std()
            if col_std == 0:  # Skip if no variance
                continue
            z_scores = np.abs((df[col] - df[col].mean()) / col_std)
            if config.handle_outliers == "clip":
                threshold = config.outlier_threshold
                df[col] = df[col].clip(
                    lower=df[col].mean() - threshold * col_std,
                    upper=df[col].mean() + threshold * col_std,
                )
            elif config.handle_outliers == "remove":
                df = df[z_scores < config.outlier_threshold]

    return df


def encode_categorical(
    df: pd.DataFrame, config: PreprocessingConfig, target_col: Optional[str] = None
) -> tuple[pd.DataFrame, Optional[ColumnTransformer]]:
    """Step 5: Categorical Encoding."""
    df = df.copy()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if target_col and target_col in categorical_cols:
        categorical_cols.remove(target_col)

    if config.encode_categorical == "none" or len(categorical_cols) == 0:
        return df, None

    transformer = None

    if config.encode_categorical == "onehot":
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop="first")
        encoded = encoder.fit_transform(df[categorical_cols])
        encoded_df = pd.DataFrame(
            encoded, columns=encoder.get_feature_names_out(categorical_cols), index=df.index
        )
        df = df.drop(columns=categorical_cols)
        df = pd.concat([df, encoded_df], axis=1)
        transformer = encoder

    elif config.encode_categorical == "label":
        le = LabelEncoder()
        for col in categorical_cols:
            df[col] = le.fit_transform(df[col].astype(str))

    return df, transformer


def scale_numerical(
    df: pd.DataFrame, config: PreprocessingConfig, target_col: Optional[str] = None
) -> tuple[pd.DataFrame, Optional[Any]]:
    """Step 6: Numerical Scaling."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)

    if config.scale_numerical == "none" or len(numeric_cols) == 0:
        return df, None

    scaler = None

    if config.scale_numerical == "standard":
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    elif config.scale_numerical == "minmax":
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    elif config.scale_numerical == "robust":
        scaler = RobustScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df, scaler


def split_train_test(
    df: pd.DataFrame,
    target_col: str,
    config: PreprocessingConfig,
    problem_type: str = "regression",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Step 7: Train-test split."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    stratify_param = None
    if config.stratify and problem_type != "regression":
        stratify_param = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify_param,
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    return train_df, test_df, y_train, y_test


def run_preprocessing_pipeline(
    df: pd.DataFrame,
    target_col: str,
    config: PreprocessingConfig,
    problem_type: str = "regression",
) -> PreprocessingResult:
    """
    Complete preprocessing pipeline.

    Steps:
    1. Data Cleaning
    2. Data Reduction
    3. Data Transformation
    4. Outlier Handling
    5. Categorical Encoding
    6. Numerical Scaling
    7. Train-test split
    """
    df = df.copy()

    summary = {
        "initial_shape": df.shape,
        "steps_applied": [],
        "columns_dropped": [],
        "columns_added": [],
    }

    # Step 1: Cleaning
    df = clean_data(df, config)
    summary["steps_applied"].append("cleaning")
    summary["after_cleaning"] = df.shape

    # Step 2: Reduction
    initial_cols = set(df.columns)
    df = reduce_data(df, config)
    dropped_cols = initial_cols - set(df.columns)
    summary["columns_dropped"].extend(list(dropped_cols))
    summary["steps_applied"].append("reduction")
    summary["after_reduction"] = df.shape

    # Step 3: Transformation
    df = transform_features(df, config)
    summary["steps_applied"].append("transformation")
    summary["after_transformation"] = df.shape

    # Step 4: Outlier Handling
    df = handle_outliers(df, config)
    summary["steps_applied"].append("outlier_handling")
    summary["after_outliers"] = df.shape

    # Step 5: Categorical Encoding
    df, cat_transformer = encode_categorical(df, config, target_col)
    summary["steps_applied"].append("categorical_encoding")
    summary["after_encoding"] = df.shape

    # Step 6: Numerical Scaling
    df, num_scaler = scale_numerical(df, config, target_col)
    summary["steps_applied"].append("numerical_scaling")
    summary["after_scaling"] = df.shape

    # Step 7: Train-test split
    train_df, test_df, y_train, y_test = split_train_test(df, target_col, config, problem_type)
    summary["steps_applied"].append("train_test_split")

    # Final feature names (excluding target)
    feature_names = [col for col in train_df.columns if col != target_col]

    return PreprocessingResult(
        train_df=train_df,
        test_df=test_df,
        feature_names=feature_names,
        target_name=target_col,
        n_features=len(feature_names),
        n_train=len(train_df),
        n_test=len(test_df),
        preprocessing_summary=summary,
    )

