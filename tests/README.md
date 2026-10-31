# Tests for Churn Project

This directory contains all test files for the Churn Project.

## Running Tests

```bash
# Install development dependencies first
pip install -r requirements-dev.txt

# Run all tests
pytest tests/

# Run with coverage report
pytest --cov=app --cov=scripts tests/

# Run specific test file
pytest tests/test_model.py

# Run tests with verbose output
pytest -v tests/

# Run tests and generate HTML coverage report
pytest --cov=app --cov=scripts --cov-report=html tests/
```

## Test Structure

```
tests/
├── __init__.py                 # Makes tests a Python package
├── conftest.py                 # Shared pytest fixtures
├── test_model.py              # Model training and prediction tests
├── test_preprocessing.py      # Data preprocessing tests
├── test_app.py                # Streamlit app integration tests
└── test_utils.py              # Utility function tests
```

## Writing New Tests

### Example Test

```python
import pytest
import pandas as pd
from scripts.train_model import preprocess_data

def test_preprocess_removes_customer_id():
    """Test that customerID column is removed during preprocessing."""
    df = pd.DataFrame({
        'customerID': ['C001', 'C002'],
        'tenure': [12, 24],
        'Churn': ['No', 'Yes']
    })
    
    processed = preprocess_data(df)
    assert 'customerID' not in processed.columns

def test_churn_mapping():
    """Test that Churn values are correctly mapped to 0/1."""
    df = pd.DataFrame({
        'Churn': ['No', 'Yes', 'No']
    })
    
    y = df['Churn'].map({'No': 0, 'Yes': 1})
    assert list(y) == [0, 1, 0]
```

### Using Fixtures

```python
# conftest.py
import pytest
import pandas as pd

@pytest.fixture
def sample_customer_data():
    """Provide sample customer data for testing."""
    return pd.DataFrame({
        'customerID': ['7590-VHVEG'],
        'gender': ['Female'],
        'SeniorCitizen': [0],
        'Partner': ['Yes'],
        'Dependents': ['No'],
        'tenure': [12],
        'MonthlyCharges': [65.5],
        'TotalCharges': [786.0],
        'Churn': ['No']
    })

# test_model.py
def test_prediction_shape(sample_customer_data, trained_model):
    """Test that prediction output has correct shape."""
    features = sample_customer_data.drop(columns=['customerID', 'Churn'])
    prediction = trained_model.predict(features)
    
    assert prediction.shape == (1,)
    assert prediction[0] in [0, 1]
```

## Test Categories

### Unit Tests
Test individual functions and components in isolation.

**Coverage:**
- Data preprocessing functions
- Feature engineering
- Model prediction logic
- Utility functions

### Integration Tests
Test interactions between components.

**Coverage:**
- Full training pipeline
- Model serialization/deserialization
- Dashboard data loading
- End-to-end prediction workflow

### Smoke Tests
Quick tests to verify basic functionality works.

**Coverage:**
- Model file loads without error
- Data file exists and is readable
- Dashboard starts successfully

## Continuous Integration

_(Coming soon)_

Tests will automatically run on:
- Pull requests
- Commits to main branch
- Scheduled daily runs

## Coverage Goals

- **Target**: >80% code coverage
- **Current**: _Not yet measured_

## Known Issues

- [ ] Add tests for Streamlit UI components
- [ ] Mock external dependencies
- [ ] Add performance/benchmark tests
- [ ] Create test fixtures for model objects

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [Python testing best practices](https://docs.python-guide.org/writing/tests/)
