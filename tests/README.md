# MLflow Test Scripts

## test1.py

This script demonstrates how to create parameterized MLflow runs that can accept command-line arguments.

### Usage

Run the test script directly with MLflow:

```bash
mlflow run . -e test1
```

Or with specific parameters:

```bash
mlflow run . -e test1 -P n_estimators=150 -P max_depth=7 -P max_features=3
```

### Parameters

- `n_estimators`: Number of trees in the random forest (default: 100)
- `max_depth`: Maximum depth of each tree (default: 5)
- `max_features`: Number of features to consider when looking for the best split (default: 5)

### Functionality

- Connects to MLflow tracking server at `http://localhost:5000`
- Loads the diabetes dataset from sklearn
- Trains a RandomForestRegressor with the specified parameters
- Logs parameters, metrics, and model artifacts to MLflow
- Calculates and logs MSE and R2 score