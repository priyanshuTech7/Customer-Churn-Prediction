# Churn Project - Architecture Overview

## System Architecture

Churn Project follows a modular, pipeline-based architecture designed for maintainability and scalability.

```
┌─────────────────────────────────────────────────────────────────────┐
│                       Churn Project System                          │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
         ┌──────────────────┐        ┌──────────────────┐
         │  Offline Training│        │  Online Serving  │
         │     Pipeline     │        │   (Streamlit)    │
         └──────────────────┘        └──────────────────┘
                    │                           │
                    │                           │
         ┌──────────┴──────────┐    ┌──────────┴──────────┐
         ▼          ▼          ▼    ▼          ▼          ▼
    ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐
    │ Data │  │Train │  │Export│  │ Load │  │ UI   │  │Predict│
    │ Load │→ │Models│→ │Model │→ │Model │→ │Render│→ │  &   │
    │      │  │      │  │      │  │      │  │      │  │Explain│
    └──────┘  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘
```

## Component Breakdown

### 1. Data Layer

**Location**: `data/`

**Responsibilities**:
- Store raw customer dataset (IBM Telco)
- Provide read access to Streamlit app and training scripts

**Key Files**:
- `WA_Fn-UseC_-Telco-Customer-Churn.csv` (7,043 records)

### 2. Training Pipeline

**Location**: `scripts/train_model.py`

**Flow**:
```python
Raw Data → Clean → Feature Engineering → Train/Test Split → 
Model Training → Evaluation → Model Export
```

**Components**:

#### 2.1 Preprocessing
```python
# Column transformer with two parallel tracks
ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(), categorical_features)
])
```

#### 2.2 Model Training
- Balanced Logistic Regression (class_weight='balanced')
- Random Forest Classifier (150 trees, depth=12)
- Decision Tree Classifier (depth=7)

#### 2.3 Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC
- Confusion Matrix
- ROC Curve data points

### 3. Model Serialization

**Format**: Joblib pickle (`.pkl`)

**Contents**:
```python
{
    'pipeline': <fitted Pipeline object>,
    'metrics': {
        'accuracy': 0.7381,
        'recall': 0.7834,
        'roc_auc': 0.8416,
        ...
    },
    'feature_names': [...],
    'model_comparison': {...}
}
```

### 4. Serving Layer (Streamlit App)

**Location**: `app/app.py` (1,563 lines)

**Architecture**:

```
app.py
├── Configuration & Styling (CSS)
├── Data Loading Module
├── Model Loading Module
├── Sidebar Navigation
└── Page Modules
    ├── Predict Tab
    │   ├── Customer Search
    │   ├── Feature Input
    │   ├── Prediction Engine
    │   └── Retention Playbook Generator
    ├── Analytics Tab
    │   ├── Cohort Filters
    │   ├── Churn Composition Donut
    │   ├── Payment Method Analysis
    │   ├── Correlation Matrix
    │   └── Scatter Plots
    ├── Model Insights Tab
    │   ├── Feature Importance
    │   ├── Confusion Matrix
    │   ├── ROC Curve
    │   └── Model Comparison
    ├── Data Explorer Tab
    │   ├── Search & Filter
    │   ├── Paginated Table
    │   └── CSV Export
    └── Summary Tab
        └── Project Overview
```

## Data Flow

### Prediction Workflow

```
User Input → Feature Vector → Preprocessor → Model → Probability →
Risk Tier Classification → Feature Attribution → Action Playbook
```

**Step Details**:

1. **User Input**: Customer ID lookup OR manual feature entry
2. **Feature Vector**: 19-feature DataFrame construction
3. **Preprocessor**: StandardScaler + OneHotEncoder (creates ~50 features)
4. **Model**: Logistic Regression prediction
5. **Probability**: Output from `predict_proba()[0][1]`
6. **Risk Tier**: Threshold-based classification
7. **Feature Attribution**: Coefficient × feature value
8. **Action Playbook**: Rule-based recommendations

### Analytics Workflow

```
Raw Data → Cohort Filters → Aggregation → Visualization → Display
```

## Key Design Patterns

### 1. Pipeline Pattern
All preprocessing and modeling encapsulated in scikit-learn Pipeline:

```python
Pipeline([
    ('preprocessor', ColumnTransformer(...)),
    ('classifier', LogisticRegression(...))
])
```

**Benefits**:
- Prevents train/test leakage
- Simplifies deployment
- Ensures reproducibility

### 2. Caching Pattern
Streamlit's `@st.cache_data` decorator for expensive operations:

```python
@st.cache_data
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)
```

**Benefits**:
- Faster page reloads
- Reduced I/O operations
- Better user experience

### 3. State Management
Streamlit session state for user interactions:

```python
if 'selected_customer' not in st.session_state:
    st.session_state.selected_customer = None
```

### 4. Modular Visualization
Reusable Plotly chart configurations:

```python
def create_donut_chart(data, title):
    fig = go.Figure(data=[go.Pie(...)])
    fig.update_layout(...)
    return fig
```

## Technology Stack

### Core Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| **Python** | 3.12 | Core language |
| **Streamlit** | 1.66.1 | Web dashboard framework |
| **scikit-learn** | 1.9.1 | ML algorithms & preprocessing |
| **Pandas** | 2.2.3 | Data manipulation |
| **NumPy** | 2.5.4 | Numerical computation |
| **Plotly** | 7.1.0 | Interactive visualizations |
| **Joblib** | 1.6.0 | Model serialization |

### Development Tools

| Tool | Purpose |
|------|---------|
| **pytest** | Unit testing |
| **black** | Code formatting |
| **flake8** | Linting |
| **mypy** | Type checking |

## Scalability Considerations

### Current Limitations
- **In-memory data**: All 7,043 records loaded into RAM
- **Single-threaded**: Streamlit runs on one thread
- **No database**: Data persists only in CSV
- **No API**: Predictions only via UI

### Future Enhancements

#### Phase 1: Database Integration
```
PostgreSQL/SQLite → Replace CSV storage
```

#### Phase 2: API Layer
```
FastAPI → REST endpoints for predictions
```

#### Phase 3: Batch Processing
```
Celery → Async batch predictions
```

#### Phase 4: Model Monitoring
```
MLflow → Track model drift & performance
```

## Security Considerations

### Current State
- No authentication
- Local file system only
- No sensitive data exposure

### Production Requirements
- OAuth2/JWT authentication
- Role-based access control (RBAC)
- Encrypted data storage
- Audit logging
- Rate limiting

## Performance Characteristics

### Latency Profile

| Operation | Latency | Notes |
|-----------|---------|-------|
| Model loading | ~500ms | One-time (cached) |
| Data loading | ~200ms | One-time (cached) |
| Single prediction | <5ms | Very fast |
| Batch 1000 predictions | ~150ms | Linear scaling |
| Dashboard render | ~1-2s | Initial load |

### Memory Footprint

| Component | Memory |
|-----------|--------|
| Streamlit process | ~150 MB |
| Pandas DataFrame | ~5 MB |
| Model object | ~10 MB |
| **Total** | **~165 MB** |

## Error Handling Strategy

### Graceful Degradation
1. Model file missing → Display error message
2. Data file missing → Show sample data placeholder
3. Invalid customer ID → Suggest valid IDs
4. Prediction error → Log and show generic message

### Logging
Currently: Print statements  
Future: Structured logging with Python `logging` module

## Testing Strategy

See [Testing Guide](../tests/README.md) for details.

### Test Pyramid
```
        ┌───────────┐
        │ E2E Tests │  ← Few (Streamlit UI)
        └───────────┘
      ┌───────────────┐
      │Integration Tests│ ← Some (Pipeline)
      └───────────────┘
   ┌────────────────────┐
   │    Unit Tests      │ ← Many (Functions)
   └────────────────────┘
```

## Deployment Architecture

### Local Development
```
Developer Machine → venv → Streamlit Dev Server → localhost:8501
```

### Docker Container (Planned)
```
Dockerfile → Image → Container → Exposed Port 8501
```

### Cloud Deployment (Planned)
```
GitHub → CI/CD → Docker Registry → 
Cloud Run/ECS/App Service → Public URL
```

## Directory Structure Philosophy

```
Customer-Churn-Prediction1/
├── app/              # Runtime serving code
├── data/             # Static datasets
├── models/           # Trained artifacts
├── scripts/          # Training & batch jobs
├── notebooks/        # Exploration & research
├── tests/            # Test suite
├── docs/             # Documentation
└── logs/             # Runtime logs (gitignored)
```

**Separation of Concerns**:
- **app/**: What users interact with
- **scripts/**: What data scientists run
- **tests/**: What CI/CD validates
- **docs/**: What developers read

---

**Last Updated**: 2026-10-31  
**Version**: 1.0.0
