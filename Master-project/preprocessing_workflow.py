"""
Preprocessing Workflow - Step-by-step preprocessing helper functions
"""

from typing import Any, Dict, List


# Step definitions
PREPROCESSING_STEPS = {
    1: {
        "name": "data_cleaning",
        "title": "Data Cleaning",
        "description": "Handle missing values and remove duplicates",
        "applies_to": ["all"],  # All problem types
    },
    2: {
        "name": "data_reduction",
        "title": "Data Reduction",
        "description": "Drop columns with high missing values or low variance",
        "applies_to": ["all"],  # All problem types (optional)
    },
    3: {
        "name": "outlier_handling",
        "title": "Outlier Handling",
        "description": "Detect and handle outliers using IQR or Z-score methods",
        "applies_to": ["classification", "regression"],  # Mainly for supervised learning
    },
    4: {
        "name": "categorical_encoding",
        "title": "Categorical Encoding",
        "description": "Encode categorical variables (one-hot, label, target encoding)",
        "applies_to": ["all"],  # All problem types with categorical features
    },
    5: {
        "name": "numerical_scaling",
        "title": "Numerical Scaling",
        "description": "Scale numerical features (StandardScaler, MinMaxScaler, RobustScaler)",
        "applies_to": ["classification", "regression"],  # Usually for non-tree models
    },
    6: {
        "name": "data_transformation",
        "title": "Data Transformation",
        "description": "Apply log or power transformations to features",
        "applies_to": ["classification", "regression"],  # Optional for supervised learning
    },
    7: {
        "name": "train_test_split",
        "title": "Train-Test Split",
        "description": "Split data into training and testing sets",
        "applies_to": ["classification", "regression"],  # NOT for clustering, anomaly detection
    },
}


def get_steps_for_problem_type(problem_type: str, problem_subtype: str | None = None) -> List[Dict[str, Any]]:
    """Get list of preprocessing steps that apply to the given problem type."""
    applicable_steps = []
    
    for step_num, step_info in PREPROCESSING_STEPS.items():
        applies_to = step_info["applies_to"]
        if "all" in applies_to or problem_type in applies_to:
            applicable_steps.append({
                "step_number": step_num,
                **step_info,
            })
    
    return applicable_steps


def get_step_info(step_number: int) -> Dict[str, Any] | None:
    """Get information about a specific preprocessing step."""
    return PREPROCESSING_STEPS.get(step_number)


def get_next_step(current_step: int, problem_type: str) -> int | None:
    """Get the next step number after current_step for the given problem type."""
    applicable_steps = get_steps_for_problem_type(problem_type)
    step_numbers = [s["step_number"] for s in applicable_steps]
    step_numbers.sort()
    
    try:
        current_index = step_numbers.index(current_step)
        if current_index + 1 < len(step_numbers):
            return step_numbers[current_index + 1]
    except ValueError:
        # Current step not in applicable steps, return first step
        return step_numbers[0] if step_numbers else None
    
    return None  # No next step (completed)


def is_step_applicable(step_number: int, problem_type: str) -> bool:
    """Check if a step is applicable to the given problem type."""
    step_info = get_step_info(step_number)
    if not step_info:
        return False
    
    applies_to = step_info["applies_to"]
    return "all" in applies_to or problem_type in applies_to
