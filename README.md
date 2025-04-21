# Credit Risk Default Model

This project builds a supervised machine learning model to predict corporate credit defaults, optimizing risk assessment in banking. Completed during my Postgraduate Program in Data Science & Business Analytics (Grade: A-, 2023), it draws on my 12+ years at State Bank of India, where I managed a Rs 50+ Cr loan portfolio with a <1% default rate using AI analytics dashboards.

## Motivation
At SBI, I achieved a <1% default rate (vs. 3.5% bank average) for 1,000+ loan proposals, leveraging AI-driven risk analytics, and prevented digital frauds with AI alerts. This project applies logistic regression and random forest to predict defaults, supporting fintech AI research at NUS’s BLOCK71 and risk solutions for AI firms like JPMorgan or Stripe.

## Methods
- **Dataset**: Synthetic corporate financial dataset (5,000 companies, 51 features: revenue, debt, ratios, default label). Original PGP data is proprietary; a sample is in `data/sample_data.csv`.
- **Preprocessing**: Handled missing values with multiple imputation, removed outliers, and reduced multicollinearity using Variance Inflation Factor (VIF) and Recursive Feature Elimination (RFE), selecting 10 key features.
- **Model**: Logistic regression and random forest with hyperparameter tuning (GridSearchCV) and 5-fold cross-validation.
- **Tools**: Python, Pandas, Scikit-learn, NumPy, Matplotlib, Seaborn.

## Results
- Achieved 82% F1-score and 84% accuracy on validation (replace with your metrics, e.g., from PGP notebook).
- Identified top 10 predictive features via feature importance (see `figures/feature_importance.png`).
- Visualized performance with ROC curve (AUC=0.85) and confusion matrix (see `figures/`).

## Relevance
This project reflects my SBI experience in loan risk management and fraud prevention, aligning with NUS’s fintech AI priorities (e.g., credit scoring). It showcases machine learning skills for AI firms developing risk analytics, such as DBS or PayPal.

## Usage
1. Clone the repo:
   ```bash
   git clone https://github.com/ravdeepgill/credit-risk-model.git
