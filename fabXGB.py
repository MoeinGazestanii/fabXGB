# fabxgb.py
#
# Frequency Adjusted Borders Ordinal XGBoost (fabXGB)
#
# Implements an ordinal regression strategy that fits a standard XGBoost
# regressor to numeric category scores, then optimizes decision thresholds
# (borders) based on Out-of-Fold (OOF) prediction quantiles to match the 
# prior distribution of the target variable.
#
# Author: Moein Gazestani
# License: MIT

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from xgboost import XGBRegressor
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class FabXGBClassifier(BaseEstimator, ClassifierMixin):
    """
    Frequency Adjusted Borders XGBoost for Ordinal Regression.
    
    Fits a regression model to numeric scores assigned to ordinal categories,
    then determines decision boundaries using the quantiles of OOF predictions
    to match the class frequencies of the training data.
    """
    def __init__(self, scores=None, n_estimators=100, max_depth=3, 
                 learning_rate=0.1, n_splits=5, n_repeats=1, 
                 random_state=42, n_jobs=-1, **kwargs):
        self.scores = scores
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.kwargs = kwargs # Catch-all for other XGB params
        
    def fit(self, X, y):
        """
        Fit the model and estimate ordinal borders.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : pandas Series or array-like
            Target variable. Must be ordered categorical.
        """
        # Validate inputs
        if isinstance(y, pd.Series):
            if not isinstance(y.dtype, CategoricalDtype) or not y.cat.ordered:
                raise ValueError("y must be an ordered pandas Categorical Series.")
            self.classes_ = y.cat.categories
            y_codes = y.cat.codes
        else:
            # Fallback for non-pandas input (assumes sorted integers 0..k)
            y = np.array(y)
            self.classes_ = np.unique(y)
            y_codes = np.searchsorted(self.classes_, y)
            
        X_chk, y_codes = check_X_y(X, y_codes)
        
        # 1. Assign Numeric Scores
        n_classes = len(self.classes_)
        if self.scores is None:
            self.scores_ = np.arange(1, n_classes + 1)
        else:
            if len(self.scores) != n_classes:
                raise ValueError(f"Length of scores ({len(self.scores)}) must match classes ({n_classes}).")
            self.scores_ = np.array(self.scores)
            
        y_numeric = self.scores_[y_codes]
        
        # 2. Fit Main Model (Regressor)
        self.regressor_ = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective="reg:squarederror",
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            **self.kwargs
        )
        self.regressor_.fit(X, y_numeric)
        
        # 3. OOF Predictions for Border Estimation
        # Adjust n_splits for small datasets
        actual_splits = min(self.n_splits, len(np.unique(y_codes)), len(y))
        
        cv = RepeatedStratifiedKFold(n_splits=actual_splits, 
                                     n_repeats=self.n_repeats, 
                                     random_state=self.random_state)
        
        oof_preds = np.zeros(len(y))
        
        # We need to handle pandas DataFrame vs numpy array for indexing
        if isinstance(X, pd.DataFrame):
            X_access = X.values
        else:
            X_access = X_chk

        for tr_idx, val_idx in cv.split(X_access, y_codes):
            # Fit temporary model
            m = XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                objective="reg:squarederror",
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                **self.kwargs
            )
            m.fit(X_access[tr_idx], y_numeric[tr_idx])
            
            # Accumulate predictions (summing up)
            oof_preds[val_idx] += m.predict(X_access[val_idx])

        # Average the accumulated predictions
        oof_preds /= self.n_repeats

        # 4. Calculate Frequency-Adjusted Borders
        # Calculate cumulative frequency of classes
        counts = np.bincount(y_codes, minlength=n_classes)
        cat_cumsum = np.cumsum(counts) / len(y)
        
        # Find quantiles in the OOF predictions that match the class frequencies
        # We only need the inner cuts (between classes)
        self.borders_ = np.quantile(oof_preds, cat_cumsum[:-1])
        
        return self

    def predict(self, X):
        """Predict class labels for X."""
        check_is_fitted(self, ['regressor_', 'borders_', 'classes_'])
        X = check_array(X)
        
        # Predict continuous scores
        preds = self.regressor_.predict(X)
        
        # Digitize places values into bins based on borders
        # bins=0 -> < border[0] -> class 0
        # bins=1 -> >= border[0], < border[1] -> class 1
        class_indices = np.digitize(preds, self.borders_)
        
        # Map back to original category labels
        return self.classes_[class_indices]

    def predict_proba(self, X):
        """
        Estimate probability (soft classification).
        Note: This is an approximation using distance to borders.
        """
        # Ordinal models don't naturally give probabilities without
        # assuming a distribution (e.g. Probit/Logit). 
        # This is a placeholder or you can implement logic here.
        raise NotImplementedError("Probability estimation not yet implemented for fabXGB.")






if __name__ == "__main__":
    from sklearn.metrics import classification_report, confusion_matrix
    
    # 1. Generate Synthetic Ordinal Data
    np.random.seed(42)
    N = 1000
    
    # Feature 1: Linear correlation with target
    X1 = np.random.normal(0, 1, N)
    # Feature 2: Noise
    X2 = np.random.normal(0, 1, N)
    
    # Latent variable (Truth)
    y_continuous = 3 * X1 + np.random.normal(0, 0.5, N)
    
    # Create ordinal bins from latent variable
    bins = np.quantile(y_continuous, [0.33, 0.66])
    y_codes = np.digitize(y_continuous, bins)
    
    # Convert to Pandas Categorical
    cat_map = {0: "Low", 1: "Medium", 2: "High"}
    y = pd.Series([cat_map[c] for c in y_codes]).astype(
        CategoricalDtype(categories=["Low", "Medium", "High"], ordered=True)
    )
    
    X = pd.DataFrame({"Feature_Signal": X1, "Feature_Noise": X2})

    # 2. Train fabXGB
    model = FabXGBClassifier(n_estimators=100, n_splits=5, n_repeats=2)
    model.fit(X, y)

    # 3. Predict
    print(f"Borders found: {model.borders_}")
    
    # Test on new data (similar distribution)
    X_new = pd.DataFrame({"Feature_Signal": [2.0, -2.0, 0.1], "Feature_Noise": [0.5, 0.5, 0.5]})
    preds = model.predict(X_new)
    
    print("\n--- Predictions for New Data ---")
    print(f"Input: 2.0 (High Signal)  -> Prediction: {preds[0]}")
    print(f"Input: -2.0 (Low Signal)  -> Prediction: {preds[1]}")
    print(f"Input: 0.1 (Ambiguous)    -> Prediction: {preds[2]}")

    # 4. Evaluation on training set
    print("\n--- Training Report ---")
    print(classification_report(y, model.predict(X)))