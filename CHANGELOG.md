# Changelog

All notable changes to Churn Project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Real-time prediction API with FastAPI
- Docker containerization
- SHAP value integration for enhanced explainability
- Automated model retraining pipeline

---

## [1.0.0] - 2026-10-31

### Added
- 🎉 **Initial release** of Churn Project
- Interactive Streamlit dashboard with 5 main sections:
  - Predict: Customer search and real-time churn scoring
  - Analytics: Cohort analysis and visualization suite
  - Model Insights: Explainability tools (ROC, confusion matrix, feature importance)
  - Data: Searchable customer database with pagination
  - Summary: Project overview and benchmarking results
- Balanced Logistic Regression model achieving:
  - 78.34% recall
  - 73.81% accuracy
  - 0.8416 ROC-AUC
- Customer database search by ID with 7,043 records
- Multi-dimensional cohort filtering (contract, service, tenure, demographics)
- Prescriptive retention action playbook
- Feature attribution with signed contributions
- Risk tier classification (Low, Moderate, High, Critical)
- Interactive Plotly visualizations:
  - Donut charts for churn composition
  - Correlation heatmaps
  - Scatter plots (tenure vs. charges)
  - ROC curves
  - Confusion matrices
- Model comparison benchmarking (Logistic Regression, Random Forest, Decision Tree)
- Dark cybernetic dashboard theme with responsive layout
- Production-style preprocessing pipeline with scikit-learn
- Comprehensive README with installation guide
- Training script with multi-model evaluation
- Jupyter notebook for exploratory data analysis

### Technical Stack
- Python 3.12
- Streamlit 1.66.1
- scikit-learn 1.9.1
- Pandas 2.2.3
- NumPy 2.5.4
- Plotly 7.1.0
- Joblib 1.6.0

### Dataset
- IBM Telco Customer Churn Dataset
- 7,043 customer records
- 21 features (demographics, services, billing)
- Binary target variable (Churn: Yes/No)

---

## Version History

### Release Naming Convention
- **Major (X.0.0)**: Breaking changes, major feature additions
- **Minor (1.X.0)**: New features, backward-compatible
- **Patch (1.0.X)**: Bug fixes, minor improvements

### Future Roadmap

#### v1.1.0 (Q1 2027)
- [ ] Add SHAP explainability integration
- [ ] Customer segmentation clustering (K-means)
- [ ] Export predictions to CSV/Excel
- [ ] Email notification system for high-risk customers
- [ ] Model performance monitoring dashboard

#### v1.2.0 (Q2 2027)
- [ ] FastAPI REST API for predictions
- [ ] Authentication system (OAuth2)
- [ ] Multi-user support with role-based access
- [ ] Historical prediction tracking database
- [ ] A/B testing framework for model versions

#### v2.0.0 (Q3 2027)
- [ ] Real-time streaming predictions (Apache Kafka)
- [ ] MLflow integration for experiment tracking
- [ ] AutoML model selection pipeline
- [ ] Multi-dataset support (SaaS, Banking, E-commerce)
- [ ] Prescriptive action reinforcement learning agent

---

## Deprecated Features

_None yet_

---

## Security Updates

_None yet_

---

## Breaking Changes

_None yet_

---

## Contributors

Special thanks to all contributors who helped build Churn Project!

- Initial development and architecture
- Dataset integration and preprocessing
- Model training and evaluation framework
- Streamlit dashboard design and implementation

---

## Links

- [GitHub Repository](https://github.com/yourusername/Customer-Churn-Prediction1)
- [Issue Tracker](https://github.com/yourusername/Customer-Churn-Prediction1/issues)
- [Documentation](https://github.com/yourusername/Customer-Churn-Prediction1/wiki)

---

[Unreleased]: https://github.com/yourusername/Customer-Churn-Prediction1/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/yourusername/Customer-Churn-Prediction1/releases/tag/v1.0.0
