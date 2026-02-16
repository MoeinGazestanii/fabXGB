# fabXGB: Frequency-Adjusted Borders XGBoost

**fabXGB** is a novel method for ordinal prediction that extends the principles of *Frequency-Adjusted Borders Ordinal Forest (fabOF)* by leveraging the scalability and predictive accuracy of **XGBoost**.

Developed as part of a Master's Thesis, this approach addresses the limitations of Random Forest-based ordinal methods by combining gradient boosting regression with a distribution-aware border estimation framework.

## 📌 Abstract

Ordinal responses—such as educational grades, healthcare severity levels, and risk assessments—require models that respect the natural ordering of categories. While modern machine learning approaches like Random Forests (fabOF) have proven effective, they often lag behind gradient boosting on structured data.

**fabXGB** solves this by:
1.  **Regression:** Fitting an XGBoost model (Chen & Guestrin, 2016) on integer-encoded ordinal labels.
2.  **OOF Border Estimation:** Deriving decision boundaries (borders) from **Out-of-Fold (OOF)** predictions to match the prior class distributions of the training data.

This design preserves the interpretability and distribution-awareness of fabOF while significantly enhancing predictive strength.

## 🚀 Key Features

* **State-of-the-Art Boosting:** Utilizes XGBoost for superior performance on structured tabular data.
* **Frequency-Adjusted Borders:** Automatically learns decision thresholds that respect class imbalance and distribution.
* **Interpretability:** Integrates permutation-based variable importance.
* **Optimization:** Tuned using Linearly Weighted Cohen’s Kappa as the primary criterion.


## 📊 Methodology

The core algorithm operates in two stages:

1.  **Scoring:** An XGBoost regressor predicts continuous scores for the input data.
2.  **Thresholding:** Decision borders are calculated using the quantiles of OOF predictions. 

## 🏆 Performance

Experiments on the **Student Performance Dataset** (Cortez and Silva, 2008) demonstrate that **fabXGB outperforms fabOF and baseline models** across key metrics:

* **Weighted Cohen's Kappa**
* **F1 Score (Macro/Weighted)**
* **Kendall’s Tau**

## 💻 Installation

```bash
git clone [https://github.com/MoeinGazestanii/fabXGB.git](https://github.com/MoeinGazestanii/fabXGB.git)
cd fabXGB
pip install -r requirements.txt
