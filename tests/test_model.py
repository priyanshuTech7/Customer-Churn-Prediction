"""Unit tests for model training and prediction."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


def test_model_file_exists(model_path):
    """Test that the trained model file exists."""
    assert model_path.exists(), f"Model file not found at {model_path}"


def test_model_loads_successfully(trained_model):
    """Test that the model can be loaded without errors."""
    assert trained_model is not None
    assert hasattr(trained_model, 'predict')
    assert hasattr(trained_model, 'predict_proba')


def test_model_prediction_shape(trained_model, sample_features):
    """Test that predictions have correct shape."""
    predictions = trained_model.predict(sample_features)

    assert predictions.shape == (len(sample_features),)
    assert all(pred in [0, 1] for pred in predictions)


def test_model_probability_range(trained_model, sample_features):
    """Test that predicted probabilities are in valid range [0, 1]."""
    probabilities = trained_model.predict_proba(sample_features)

    assert probabilities.shape == (len(sample_features), 2)
    assert np.all(probabilities >= 0) and np.all(probabilities <= 1)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_model_feature_requirements(trained_model):
    """Test that model requires expected number of features."""
    if hasattr(trained_model, 'named_steps'):
        preprocessor = trained_model.named_steps.get('preprocessor')
        if preprocessor is not None:
            assert hasattr(preprocessor, 'transformers_')


def test_prediction_consistency(trained_model, sample_features):
    """Test that predictions are consistent across multiple calls."""
    pred1 = trained_model.predict(sample_features)
    pred2 = trained_model.predict(sample_features)

    assert np.array_equal(pred1, pred2), "Predictions should be deterministic"


def test_high_risk_customer_detected(trained_model):
    """Test that model correctly identifies high-risk customer profile."""
    high_risk = pd.DataFrame({
        'gender': ['Male'],
        'SeniorCitizen': [0],
        'Partner': ['No'],
        'Dependents': ['No'],
        'tenure': [1],
        'PhoneService': ['Yes'],
        'MultipleLines': ['No'],
        'InternetService': ['Fiber optic'],
        'OnlineSecurity': ['No'],
        'OnlineBackup': ['No'],
        'DeviceProtection': ['No'],
        'TechSupport': ['No'],
        'StreamingTV': ['No'],
        'StreamingMovies': ['No'],
        'Contract': ['Month-to-month'],
        'PaperlessBilling': ['Yes'],
        'PaymentMethod': ['Electronic check'],
        'MonthlyCharges': [85.0],
        'TotalCharges': [85.0]
    })

    proba = trained_model.predict_proba(high_risk)[0][1]
    assert proba > 0.4, f"High-risk profile should have >40% churn probability, got {proba:.2%}"


def test_low_risk_customer_detected(trained_model):
    """Test that model correctly identifies low-risk customer profile."""
    low_risk = pd.DataFrame({
        'gender': ['Female'],
        'SeniorCitizen': [0],
        'Partner': ['Yes'],
        'Dependents': ['Yes'],
        'tenure': [72],
        'PhoneService': ['Yes'],
        'MultipleLines': ['Yes'],
        'InternetService': ['DSL'],
        'OnlineSecurity': ['Yes'],
        'OnlineBackup': ['Yes'],
        'DeviceProtection': ['Yes'],
        'TechSupport': ['Yes'],
        'StreamingTV': ['Yes'],
        'StreamingMovies': ['Yes'],
        'Contract': ['Two year'],
        'PaperlessBilling': ['No'],
        'PaymentMethod': ['Credit card (automatic)'],
        'MonthlyCharges': [55.0],
        'TotalCharges': [3960.0]
    })

    proba = trained_model.predict_proba(low_risk)[0][1]
    assert proba < 0.3, f"Low-risk profile should have <30% churn probability, got {proba:.2%}"


def test_model_handles_missing_values(trained_model):
    """Test that model handles edge cases in data."""
    edge_case = pd.DataFrame({
        'gender': ['Female'],
        'SeniorCitizen': [0],
        'Partner': ['No'],
        'Dependents': ['No'],
        'tenure': [0],
        'PhoneService': ['No'],
        'MultipleLines': ['No phone service'],
        'InternetService': ['No'],
        'OnlineSecurity': ['No internet service'],
        'OnlineBackup': ['No internet service'],
        'DeviceProtection': ['No internet service'],
        'TechSupport': ['No internet service'],
        'StreamingTV': ['No internet service'],
        'StreamingMovies': ['No internet service'],
        'Contract': ['Month-to-month'],
        'PaperlessBilling': ['Yes'],
        'PaymentMethod': ['Electronic check'],
        'MonthlyCharges': [20.0],
        'TotalCharges': [0.0]
    })

    prediction = trained_model.predict(edge_case)
    proba = trained_model.predict_proba(edge_case)

    assert prediction is not None
    assert proba is not None
    assert proba.shape == (1, 2)


@pytest.mark.parametrize("contract_type,expected_risk", [
    ("Month-to-month", "high"),
    ("One year", "medium"),
    ("Two year", "low")
])
def test_contract_impact_on_churn(trained_model, contract_type, expected_risk):
    """Test that contract type appropriately affects churn probability."""
    base_customer = pd.DataFrame({
        'gender': ['Male'],
        'SeniorCitizen': [0],
        'Partner': ['Yes'],
        'Dependents': ['No'],
        'tenure': [24],
        'PhoneService': ['Yes'],
        'MultipleLines': ['No'],
        'InternetService': ['Fiber optic'],
        'OnlineSecurity': ['No'],
        'OnlineBackup': ['No'],
        'DeviceProtection': ['No'],
        'TechSupport': ['No'],
        'StreamingTV': ['No'],
        'StreamingMovies': ['No'],
        'Contract': [contract_type],
        'PaperlessBilling': ['Yes'],
        'PaymentMethod': ['Bank transfer (automatic)'],
        'MonthlyCharges': [70.0],
        'TotalCharges': [1680.0]
    })

    proba = trained_model.predict_proba(base_customer)[0][1]

    if expected_risk == "high":
        assert proba > 0.30, f"Month-to-month should be high risk, got {proba:.2%}"
    elif expected_risk == "medium":
        assert 0.15 < proba < 0.45, f"One year should be medium risk, got {proba:.2%}"
    elif expected_risk == "low":
        assert proba < 0.25, f"Two year should be low risk, got {proba:.2%}"
