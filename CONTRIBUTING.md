# Contributing to Churn Project

First off, thank you for considering contributing to Churn Project! 🎉

## How Can I Contribute?

### 🐛 Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- **Clear title** and description
- **Steps to reproduce** the behavior
- **Expected** vs **actual** behavior
- **Screenshots** if applicable
- **Environment details**: OS, Python version, browser

**Example Bug Report:**
```markdown
**Title**: Model fails to load on Python 3.11

**Description**: When running `streamlit run app/app.py` on Python 3.11.5, 
the app crashes with a joblib deserialization error.

**Steps to Reproduce**:
1. Install Python 3.11.5
2. Run `pip install -r requirements.txt`
3. Execute `streamlit run app/app.py`

**Expected**: Dashboard loads successfully
**Actual**: Error: "ValueError: unsupported pickle protocol"

**Environment**:
- OS: Windows 11
- Python: 3.11.5
- streamlit: 1.66.1
```

### 💡 Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. Provide:

- **Use case**: Why is this enhancement needed?
- **Proposed solution**: How should it work?
- **Alternatives considered**: What other approaches did you think about?

### 🔧 Pull Requests

1. **Fork** the repository
2. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** with clear commit messages
4. **Test** your changes thoroughly
5. **Update documentation** if needed
6. **Submit a PR** with a clear description

## Development Setup

### Prerequisites
- Python 3.12+
- Git
- Virtual environment tool

### Local Setup

```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/Customer-Churn-Prediction1.git
cd Customer-Churn-Prediction1

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run the app
streamlit run app/app.py
```

## Code Style Guidelines

### Python Code
- Follow **PEP 8** style guide
- Use **type hints** where appropriate
- Maximum line length: **88 characters** (Black default)
- Use **descriptive variable names**

**Good:**
```python
def calculate_churn_probability(customer_features: pd.DataFrame) -> float:
    """Calculate churn probability for a customer.
    
    Args:
        customer_features: DataFrame with customer attributes
        
    Returns:
        Churn probability between 0 and 1
    """
    prediction = model.predict_proba(customer_features)[0][1]
    return round(prediction, 4)
```

**Avoid:**
```python
def calc(x):
    return model.predict_proba(x)[0][1]
```

### Commit Messages

Use the **Conventional Commits** format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples:**
```bash
feat(predict): add customer search by phone number

Added search functionality to allow looking up customers
by their phone number in addition to customer ID.

Closes #42
```

```bash
fix(analytics): correct tenure distribution calculation

The tenure bins were miscalculated due to inclusive boundaries.
Changed to use pd.cut with right=False parameter.

Fixes #38
```

## Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=app --cov=scripts tests/

# Run specific test file
pytest tests/test_model.py
```

### Writing Tests

Place tests in the `tests/` directory:

```python
# tests/test_model.py
import pytest
import pandas as pd
from scripts.train_model import preprocess_data

def test_preprocess_handles_missing_total_charges():
    """Test that preprocessing fills missing TotalCharges with 0."""
    df = pd.DataFrame({
        'customerID': ['C001'],
        'TotalCharges': [''],
        'Churn': ['No']
    })
    
    processed = preprocess_data(df)
    assert processed['TotalCharges'].iloc[0] == 0.0
```

## Documentation

### Docstring Format

Use **Google-style docstrings**:

```python
def predict_churn(
    model: Pipeline,
    customer_data: pd.DataFrame,
    threshold: float = 0.5
) -> tuple[int, float]:
    """Predict customer churn.

    Args:
        model: Trained scikit-learn pipeline
        customer_data: Single-row DataFrame with customer features
        threshold: Classification threshold (default: 0.5)

    Returns:
        Tuple of (predicted_class, probability) where predicted_class
        is 0 (no churn) or 1 (churn), and probability is between 0-1.

    Raises:
        ValueError: If customer_data has incorrect shape or missing columns.

    Example:
        >>> data = pd.DataFrame({'tenure': [12], 'MonthlyCharges': [65.5]})
        >>> prediction, prob = predict_churn(model, data)
        >>> print(f"Churn: {prediction}, Probability: {prob:.2%}")
        Churn: 1, Probability: 78.34%
    """
    # Implementation
```

### Updating Documentation

When adding features, update:
- ✅ **README.md**: User-facing feature description
- ✅ **Inline comments**: Complex logic explanations
- ✅ **Docstrings**: Function/class documentation
- ✅ **CHANGELOG.md**: Version history entry

## Project Structure for New Files

```
Customer-Churn-Prediction1/
├── app/
│   ├── app.py              # Main Streamlit app
│   ├── components/         # NEW: Reusable UI components
│   ├── utils/              # NEW: Helper functions
│   └── config.py           # NEW: Configuration constants
│
├── scripts/
│   ├── train_model.py      # Model training script
│   ├── evaluate.py         # NEW: Model evaluation utilities
│   └── preprocessing.py    # NEW: Data preprocessing functions
│
├── tests/
│   ├── test_model.py       # Model tests
│   ├── test_app.py         # App integration tests
│   └── conftest.py         # Pytest fixtures
│
└── docs/
    ├── api.md              # API documentation
    ├── deployment.md       # Deployment guide
    └── architecture.md     # System architecture
```

## Areas We Need Help With

### High Priority
- [ ] Add unit tests for model training pipeline
- [ ] Create Dockerfile for containerized deployment
- [ ] Add SHAP values for better model explainability
- [ ] Implement A/B testing framework for model versions

### Medium Priority
- [ ] Add support for real-time prediction API (FastAPI)
- [ ] Create data validation with Great Expectations
- [ ] Add model monitoring dashboard (MLflow)
- [ ] Implement automated retraining pipeline

### Nice to Have
- [ ] Multi-language support (i18n)
- [ ] Dark/light theme toggle
- [ ] Export predictions to PDF report
- [ ] Integration with CRM systems (Salesforce, HubSpot)

## Code Review Process

All submissions require review. We use GitHub Pull Requests for this purpose.

**Reviewers will check:**
- ✅ Code quality and style
- ✅ Test coverage
- ✅ Documentation completeness
- ✅ Breaking changes
- ✅ Performance impact

## Community Guidelines

### Be Respectful
- Use welcoming and inclusive language
- Respect differing viewpoints and experiences
- Accept constructive criticism gracefully

### Be Collaborative
- Help others learn and grow
- Share knowledge generously
- Credit others' contributions

### Be Professional
- Focus on the code, not the person
- Keep discussions technical and relevant
- Assume positive intent

## Questions?

- 💬 Open a [GitHub Discussion](https://github.com/yourusername/Customer-Churn-Prediction1/discussions)
- 📧 Email: contributors@churnproject.example.com
- 📚 Check the [Wiki](https://github.com/yourusername/Customer-Churn-Prediction1/wiki)

---

**Thank you for making Churn Project better!** 🚀
