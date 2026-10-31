"""Shared pytest fixtures for Churn Project tests."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import joblib


@pytest.fixture
def sample_customer_data():
    """Provide sample customer data for testing.

    Returns:
        pd.DataFrame: Sample customer records with all features
    """
    return pd.DataFrame({
        'customerID': ['7590-VHVEG', '5575-GNVDE', '3668-QPYBK'],
        'gender': ['Female', 'Male', 'Male'],
        'SeniorCitizen': [0, 0, 1],
        'Partner': ['Yes', 'No', 'Yes'],
        'Dependents': ['No', 'No', 'No'],
        'tenure': [12, 34, 2],
        'PhoneService': ['Yes', 'Yes', 'Yes'],
        'MultipleLines': ['No', 'Yes', 'No'],
        'InternetService': ['Fiber optic', 'DSL', 'Fiber optic'],
        'OnlineSecurity': ['No', 'Yes', 'No'],
        'OnlineBackup': ['Yes', 'No', 'No'],
        'DeviceProtection': ['No', 'Yes', 'No'],
        'TechSupport': ['No', 'No', 'No'],
        'StreamingTV': ['No', 'Yes', 'No'],
        'StreamingMovies': ['No', 'No', 'No'],
        'Contract': ['Month-to-month', 'One year', 'Month-to-month'],
        'PaperlessBilling': ['Yes', 'No', 'Yes'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Electronic check'],
        'MonthlyCharges': [65.5, 56.95, 53.85],
        'TotalCharges': [786.0, 1889.5, 108.15],
        'Churn': ['No', 'No', 'Yes']
    })


@pytest.fixture
def sample_features():
    """Provide sample feature matrix for prediction.

    Returns:
        pd.DataFrame: Features without target variable
    """
    data = sample_customer_data()
    return data.drop(columns=['customerID', 'Churn'])


@pytest.fixture
def sample_target():
    """Provide sample target variable.

    Returns:
        pd.Series: Binary churn labels (0/1)
    """
    data = sample_customer_data()
    return data['Churn'].map({'No': 0, 'Yes': 1})


@pytest.fixture
def temp_data_path(tmp_path):
    """Create temporary CSV file with sample data.

    Args:
        tmp_path: pytest's temporary directory fixture

    Returns:
        Path: Path to temporary CSV file
    """
    data = sample_customer_data()
    csv_path = tmp_path / "test_data.csv"
    data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def base_dir():
    """Provide project base directory path.

    Returns:
        Path: Absolute path to project root
    """
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def model_path(base_dir):
    """Provide path to trained model file.

    Returns:
        Path: Absolute path to model pickle file
    """
    return base_dir / "models" / "customer_churn_model.pkl"


@pytest.fixture
def trained_model(model_path):
    """Load the trained model.

    Args:
        model_path: Path to model file

    Returns:
        Pipeline: Trained scikit-learn pipeline

    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    if not model_path.exists():
        pytest.skip(f"Model file not found at {model_path}")

    return joblib.load(model_path)


@pytest.fixture
def mock_streamlit_session(monkeypatch):
    """Mock Streamlit session state for testing.

    Args:
        monkeypatch: pytest monkeypatch fixture

    Yields:
        dict: Mock session state dictionary
    """
    session_state = {}

    class MockSessionState:
        def __init__(self):
            self._state = session_state

        def __getattr__(self, item):
            return self._state.get(item)

        def __setattr__(self, item, value):
            if item == '_state':
                super().__setattr__(item, value)
            else:
                self._state[item] = value

    monkeypatch.setattr("streamlit.session_state", MockSessionState())
    yield session_state


@pytest.fixture
def churn_rates():
    """Provide expected churn rates by category.

    Returns:
        dict: Known churn rates for validation
    """
    return {
        'overall': 0.2656,
        'contract': {
            'Month-to-month': 0.4270,
            'One year': 0.1130,
            'Two year': 0.0285
        },
        'payment_method': {
            'Electronic check': 0.4540,
            'Mailed check': 0.1910,
            'Bank transfer (automatic)': 0.1670,
            'Credit card (automatic)': 0.1520
        }
    }
