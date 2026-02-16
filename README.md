# fabXGB

Frequency Adjusted Borders Ordinal XGBoost (fabXGB)

fabXGB is an ordinal classification method that:
- Assigns numeric scores to ordered categories
- Fits an XGBoost regression model
- Estimates decision borders using Out-of-Fold (OOF) predictions
- Uses frequency-adjusted quantile thresholds

## Installation

```bash
git clone https://github.com/MoeinGazestanii/fabXGB.git
cd fabXGB
pip install -r requirements.txt


## Usage

After cloning the repository, you can use fabXGB as follows:

```python
from fabxgb import FabXGBClassifier
