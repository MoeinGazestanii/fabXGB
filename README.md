# fabXGB: Frequency-Adjusted Borders XGBoost

**fabXGB** is an ordinal classification method that combines gradient boosting regression with frequency-adjusted border estimation. 

The method assigns numeric scores to ordered categories, fits an XGBoost regression model, and determines decision thresholds using Out-of-Fold (OOF) predictions to match the empirical class distribution.

This implementation was developed as part of a Master's thesis in Data Science.

---

## 📌 Method Overview

The core algorithm works in two stages:

1. **Scoring:** An XGBoost regressor predicts continuous numeric scores for the input data.  
2. **Thresholding:** Decision borders are calculated using the quantiles of OOF predictions to reflect the distribution of categories in the training set.

This preserves the ordinal structure of the target variable while leveraging the predictive power of XGBoost.

---

## 💻 Installation

```bash
git clone https://github.com/MoeinGazestanii/fabXGB.git
cd fabXGB
pip install -r requirements.txt
