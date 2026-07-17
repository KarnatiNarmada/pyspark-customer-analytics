# PySpark Customer Churn Analytics — 3M+ Records

End-to-end customer churn prediction pipeline processing **3,000,000+ records** using PySpark for distributed computing. Compares three ML models with 25+ engineered features.

## Business Problem

Customer churn costs companies 5-7x more than retention. This pipeline predicts which customers are likely to churn so businesses can intervene with targeted retention strategies.

## Pipeline Architecture

```
Raw Data (3M records)
    → Load with PySpark (distributed)
    → Exploratory Data Analysis
    → Feature Engineering (12 new features)
    → ML Data Preparation (Pipeline)
    → Model Training & Evaluation (3 models)
    → Visualization & Business Insights
```

## Results

| Model | ROC-AUC | Accuracy | F1 Score | Time |
|---|---|---|---|---|
| Logistic Regression | 0.733 | 0.688 | 0.689 | 235s |
| Random Forest | 0.734 | 0.688 | 0.651 | 479s |
| **Gradient Boosted Trees** | **0.734** | **0.688** | **0.676** | **529s** |

Best Model: Gradient Boosted Trees (ROC-AUC: 0.734)

## Top Feature Importances

```
is_inactive (90+ days)         ████████████████████████  0.249
days_since_last_purchase       ██████████████████        0.184
is_low_rating (≤2 stars)       ███████████████           0.154
rating                         ██████████████            0.141
returns                        ███████████               0.110
is_high_returner (>5 returns)  ████████                  0.090
```

Business Insight: Customer inactivity and low ratings are the strongest churn predictors. Target re-engagement campaigns at customers inactive for 60+ days.

## Features Engineered

| Feature | Logic | Meaning |
|---|---|---|
| spend_per_item | amount / quantity | Purchase quality |
| return_rate | returns / purchases | Satisfaction proxy |
| is_inactive | days > 90 | At-risk flag |
| is_low_rating | rating ≤ 2 | Unhappy flag |
| is_high_spender | amount > $100 | High-value flag |
| is_frequent_buyer | purchases > 20 | Loyalty flag |
| is_high_returner | returns > 5 | Problem flag |
| engagement_score | Composite metric | Overall engagement |
| log_purchase | log(amount) | Normalized spend |
| purchase_year | Year extracted | Temporal trend |
| purchase_month | Month extracted | Seasonality |
| purchase_day | Day of week | Weekly pattern |

## Tech Stack

- **PySpark** — Distributed data processing (3M+ records)
- **Spark ML** — Logistic Regression, Random Forest, Gradient Boosted Trees
- **Python** — Pandas, NumPy, Matplotlib, Seaborn
- **ML Pipeline** — StringIndexer, VectorAssembler, StandardScaler

## Quick Start

```bash
# Clone
git clone https://github.com/KarnatiNarmada/pyspark-customer-analytics.git
cd pyspark-customer-analytics

# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Requires Java 17+
java -version

# Generate 3M records
python3 generate_data.py

# Run pipeline
python3 main.py
```

## Project Structure

```
pyspark-customer-analytics/
├── main.py              # Full PySpark ML pipeline
├── generate_data.py     # Generates 3M synthetic records
├── requirements.txt     # Dependencies
├── .gitignore
├── data/
│   └── model_results.png
└── README.md
```

## What This Demonstrates

- **Big Data:** 3,000,000+ records processed with PySpark
- **Feature Engineering:** 12 domain-specific features
- **Model Comparison:** 3 ML architectures evaluated
- **Business Impact:** Actionable churn insights
- **Production Ready:** Scalable pipeline, not just a notebook

## Author

**Narmada Karnati** — MS Data Science, Kent State University (GPA: 3.76)

[GitHub](https://github.com/KarnatiNarmada) | [LinkedIn](https://www.linkedin.com/in/narmada-karnati-b7b90a190)
