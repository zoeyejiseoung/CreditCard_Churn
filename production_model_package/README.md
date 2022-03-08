# Productionized Credit Card Customers Churn GBM Model Package

## Run With Tox (Recommended)
- Download the data from: https://www.kaggle.com/sakshigoyal7/credit-card-customers
- Save the file as `raw.csv` in the classification_model/datasets directory
- `pip install tox`
- Make sure you are in the right directory (where the tox.ini file is) then run the command: `tox` (this runs the tests and typechecks, trains the model under the hood). The first time you run this it creates a virtual env and installs
dependencies, so takes a few minutes.

## Run Without Tox
- Download the data from: https://www.kaggle.com/sakshigoyal7/credit-card-customers
- Save the file as `raw.csv` in the classification_model/datasets directory
- Add the right working directory *and* classification_model paths to your system PYTHONPATH
- `pip install -r requirements/test_requirements`
- Train the model: `python classification_model/train_pipeline.py`
- Run the tests `pytest tests`


