<div align="center">

# ◈ Churn Project

### *Predict. Retain. Prosper.*

**An intelligent ML system that transforms customer data into actionable retention strategies**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-Academic-green?style=for-the-badge)]()

[Features](#-what-makes-churn-project-special) • [Demo](#-live-demo) • [Installation](#-quick-start) • [Architecture](#-architecture) • [Performance](#-model-performance)

</div>

---

## 🎯 The Problem

Customer churn costs telecom companies **billions annually**. Traditional approaches react too late—after customers have already left. 

**Churn Project** flips the script: it predicts churn *before it happens* and prescribes personalized retention strategies for each at-risk customer.

---

## ✨ What Makes Churn Project Special

<table>
<tr>
<td width="50%">

### 🔍 **Smart Customer Search**
- Search 7,043+ customers by ID instantly
- Filter by contract, service, and actual churn status
- Compare predictions vs. historical outcomes
- Validate model accuracy in real-time

</td>
<td width="50%">

### 🎯 **Prescriptive Action Engine**
- Not just predictions—*actionable playbooks*
- Tailored retention strategies per customer
- Feature attribution shows *why* they might churn
- Risk-tiered interventions (Low → Critical)

</td>
</tr>
<tr>
<td width="50%">

### 📊 **Cohort Analytics Suite**
- Multi-dimensional filtering (contract, tenure, demographics)
- Interactive correlation heatmaps
- Payment vulnerability analysis
- Tenure vs. spending scatter plots

</td>
<td width="50%">

### 🧠 **Model Explainability**
- Global feature importance with odds ratios
- Interactive confusion matrix explorer
- ROC curve visualization (AUC: 0.8416)
- Multi-model benchmark comparison

</td>
</tr>
</table>

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- 500MB free disk space
- Modern web browser

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/Customer-Churn-Prediction1.git
cd Customer-Churn-Prediction1

# 2. Create virtual environment
python -m venv venv

# 3. Activate environment
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# 4. Install dependencies
pip install -r requirements.txt
```

### 🎬 Launch the Dashboard

```bash
streamlit run app/app.py
```

🌐 Open your browser at **`http://localhost:8501`**

---

## 💡 Live Demo

### **Predict Tab** — Real-Time Churn Scoring
```
┌─────────────────────────────────────────────────┐
│ Customer ID: 7590-VHVEG                         │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│ Churn Probability: ████████░░ 78%              │
│ Risk Level: 🔴 HIGH                             │
│                                                 │
│ Top Risk Factors:                               │
│  ⚠️  Month-to-month contract (+42%)            │
│  ⚠️  Electronic check payment (+28%)           │
│  ⚠️  Low tenure (2 months) (+19%)              │
│                                                 │
│ 📋 Recommended Actions:                         │
│  → Offer annual contract with 15% discount     │
│  → Set up automatic bank transfer              │
│  → Assign dedicated success manager            │
└─────────────────────────────────────────────────┘
```

### **Analytics Tab** — Cohort Intelligence
- **Contract Analysis**: Visualize churn rates across Monthly/Yearly/2-Year contracts
- **Payment Vulnerabilities**: Identify high-risk payment methods (electronic check = 45% churn!)
- **Tenure Distribution**: Discover the critical first 6-month retention window
- **Service Correlation Matrix**: Uncover hidden patterns in service bundles

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Churn Project Pipeline                     │
└─────────────────────────────────────────────────────────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          ▼                       ▼                       ▼
    ┌──────────┐          ┌─────────────┐         ┌──────────┐
    │   Data   │          │ Preprocessing│         │  Model   │
    │ Ingestion│  ──────► │   Pipeline   │  ─────► │ Training │
    └──────────┘          └─────────────┘         └──────────┘
    7,043 rows                    │                      │
    21 features           • One-Hot Encoding    • Logistic Reg.
                          • Standard Scaling    • Random Forest
                          • Missing value fill  • Decision Tree
                                  │                      │
                                  └──────────┬───────────┘
                                             ▼
                                    ┌─────────────────┐
                                    │  Streamlit App  │
                                    │   Dashboard     │
                                    └─────────────────┘
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    ▼                        ▼                        ▼
              ┌──────────┐            ┌──────────┐            ┌──────────┐
              │ Predict  │            │Analytics │            │  Model   │
              │   Tab    │            │   Tab    │            │ Insights │
              └──────────┘            └──────────┘            └──────────┘
```

---

## 📈 Model Performance

We trained and evaluated three algorithms on an **80/20 stratified split** of 7,043 customer records from the IBM Telco dataset:

| Algorithm | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Status |
|:----------|:--------:|:---------:|:------:|:--------:|:-------:|:------:|
| **Balanced Logistic Regression** | **73.81%** | 50.43% | **78.34%** | **61.36%** | **0.8416** | ✅ **Deployed** |
| Random Forest (150 trees) | 76.01% | 54.22% | 71.92% | 61.85% | 0.8313 | ⚪ Candidate |
| Decision Tree (depth=7) | 73.59% | 50.18% | 75.40% | 60.28% | 0.8110 | ⚪ Candidate |

### 🏆 Why Logistic Regression Won

**1. Maximum Recall (78.34%)**  
In customer retention, missing a churner costs **5-25x** more than a false alarm. Our model catches 78 out of 100 at-risk customers.

**2. Best Discrimination (ROC-AUC: 0.8416)**  
Highest area under the curve = superior class separation across all thresholds.

**3. Interpretability = Action**  
Linear coefficients translate directly to odds ratios. Marketing teams can *see* why a customer is flagged and *know* what to fix.

**4. Production-Ready**  
Lightweight, fast inference (<5ms), and stable predictions without hyperparameter drift.

---

## 📂 Project Structure

```
Customer-Churn-Prediction1/
│
├── 📱 app/
│   ├── app.py                    # Streamlit dashboard (1,563 lines)
│   └── __pycache__/              # Python bytecode cache
│
├── 📊 data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv  # IBM Telco dataset (7,043 records)
│
├── 🧠 models/
│   └── customer_churn_model.pkl  # Trained pipeline with embedded metrics
│
├── 📓 notebooks/
│   └── churn_analysis.ipynb      # Exploratory data analysis & visualization
│
├── 🔧 scripts/
│   └── train_model.py            # Model training & benchmarking script (169 lines)
│
├── 📦 venv/                      # Python virtual environment
│
├── 📄 .gitignore                 # Git exclusions
├── 📋 requirements.txt           # Python dependencies
└── 📖 README.md                  # This file
```

---

## 🔬 Key Insights from the Data

### 1. **Contract Type is King**
- Month-to-month: **42% churn rate** 🔴
- One-year: **11% churn rate** 🟡
- Two-year: **3% churn rate** 🟢

**Action**: Incentivize long-term contracts with graduated discounts.

### 2. **The Payment Method Trap**
- Electronic check users churn at **45%** (vs. 15-18% for other methods)

**Action**: Migrate e-check users to automatic bank transfers with a $10 credit.

### 3. **The First-6-Months Crisis**
- 50% of churners leave within their first 6 months
- Customers past 12 months show <20% churn probability

**Action**: Intensive onboarding + success manager touchpoints in months 1-6.

### 4. **Service Bundling Saves Customers**
- Customers with 0-1 services: **35% churn**
- Customers with 4+ services: **12% churn**

**Action**: Cross-sell complementary services during onboarding.

---

## 🛠️ Advanced Usage

### Re-train the Model

```bash
# Run the training pipeline with benchmark comparisons
python scripts/train_model.py
```

This will:
1. Load and preprocess the dataset
2. Train Logistic Regression, Random Forest, and Decision Tree
3. Evaluate all models on test set
4. Export the best model to `models/customer_churn_model.pkl`
5. Print comparison metrics

### Customize Risk Thresholds

Edit `app/app.py` lines 200-205 to adjust risk tier boundaries:

```python
# Current thresholds
LOW      = 0.30  # 0-30%
MODERATE = 0.55  # 30-55%
HIGH     = 0.75  # 55-75%
CRITICAL = 1.00  # 75-100%
```

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

- 🐛 **Bug Reports**: Open an issue with reproduction steps
- 💡 **Feature Requests**: Describe your use case and proposed solution
- 🔧 **Pull Requests**: Fork, branch, commit, and submit for review
- 📖 **Documentation**: Improve guides, add examples, fix typos

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8 mypy

# Run tests (coming soon)
pytest tests/

# Format code
black app/ scripts/

# Lint
flake8 app/ scripts/
```

---

## 📊 Technology Stack

| Layer | Technology | Purpose |
|:------|:-----------|:--------|
| **Frontend** | Streamlit 1.66 | Interactive dashboard UI |
| **Data Processing** | Pandas 2.2.3, NumPy 2.5.4 | Data manipulation & computation |
| **ML Framework** | scikit-learn 1.9.1 | Model training & evaluation |
| **Visualization** | Plotly 7.1.0 | Interactive charts & graphs |
| **Serialization** | Joblib 1.6.0 | Model persistence |
| **Language** | Python 3.12 | Core development |

---

## 🎓 Learning Resources

Want to understand the ML concepts behind Churn Project?

- [Understanding Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [ROC Curves Explained](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)
- [Handling Imbalanced Classes](https://imbalanced-learn.org/stable/)
- [SHAP Values for Model Explanation](https://shap.readthedocs.io/)

---

## 📜 License

This project is licensed for **Academic Use Only**.

For commercial licensing inquiries, please contact the project maintainers.

---

## 🙏 Acknowledgments

- **Dataset**: [IBM Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
- **Inspiration**: Real-world telecom retention challenges
- **Community**: scikit-learn, Streamlit, and Python ML communities

---

## 📞 Support

- 📧 **Email**: support@churnproject.example.com
- 💬 **Issues**: [GitHub Issues](https://github.com/yourusername/Customer-Churn-Prediction1/issues)
- 📚 **Docs**: [Wiki](https://github.com/yourusername/Customer-Churn-Prediction1/wiki)

---

<div align="center">

**Built with ❤️ for data-driven customer success**

⭐ **Star this repo** if Churn Project helped your project!

[⬆ Back to Top](#-churn-project)

</div>
