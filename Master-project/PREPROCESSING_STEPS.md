# Preprocessing Steps - Step-by-Step Workflow

## Overview
Preprocessing will proceed step-by-step based on the selected problem type. Different problem types require different preprocessing steps.

## Preprocessing Steps (Order of Execution)

### Step 1: Data Cleaning
- Handle missing values (mean/median/mode/drop/forward_fill)
- Remove duplicates
- **Applies to**: All problem types

### Step 2: Data Reduction
- Drop columns with high missing values (>50% threshold)
- Drop low variance columns
- **Applies to**: All problem types (optional)

### Step 3: Outlier Handling
- IQR method (clip/remove)
- Z-score method (clip/remove)
- **Applies to**: Regression, Classification (optional for others)

### Step 4: Categorical Encoding
- One-hot encoding
- Label encoding
- Target encoding (for classification)
- **Applies to**: All problem types with categorical features

### Step 5: Numerical Scaling
- StandardScaler
- MinMaxScaler
- RobustScaler
- None (for tree-based models)
- **Applies to**: Regression, Classification (usually not needed for tree models)

### Step 6: Data Transformation
- Log transform
- Power transform
- **Applies to**: Regression, Classification (optional)

### Step 7: Train-Test Split
- Stratified split (classification)
- Random split (regression)
- No split (clustering, anomaly detection)
- **Applies to**: Regression, Classification (NOT for clustering, anomaly detection)

## Problem Type Specific Rules

### Classification & Regression:
- All 7 steps applicable
- Train-test split: Stratified for classification, random for regression

### Clustering:
- Steps 1-6 applicable
- NO train-test split (unsupervised learning)

### Anomaly Detection:
- Steps 1-4, 6 applicable
- NO train-test split (all data used for detection)

### Dimensionality Reduction:
- Steps 1-4 applicable
- NO train-test split
- Scaling important

### Recommendation Systems:
- Steps 1-4, 6 applicable
- Custom split logic (user-item interactions)

### Time Series:
- Steps 1-4, 6 applicable
- Time-based split (not random)
- Forward fill recommended for missing values
