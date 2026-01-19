import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import numpy as np
import os

print('GridSearch Code Running....')

# Set up MLflow tracking
os.environ['MLFLOW_TRACKING_USERNAME'] = "admin"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "admin0123456789"
mlflow.set_tracking_uri("http://localhost:5000")

# Enable autologging for all supported libraries
mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True, log_models=True)

# Load dataset
db = load_diabetes()
X = db.data
y = db.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Start an MLflow run
with mlflow.start_run(run_name="gridsearch_cv_example") as run:
    print("Starting GridSearchCV with MLflow autologging...")
    
    # Define the model to tune
    rf = RandomForestRegressor(random_state=42)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Create GridSearchCV object
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,  # 5-fold cross-validation
        scoring='neg_mean_squared_error',  # Metric to optimize
        n_jobs=-1,  # Use all available cores
        verbose=1  # Show progress
    )
    
    # Fit the grid search (this will trigger autologging)
    grid_search.fit(X_train, y_train)
    
    # Log the best parameters and score
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {-grid_search.best_score_:.4f}")
    
    # Make predictions with the best model
    best_predictions = grid_search.predict(X_test)
    
    # Calculate and log test metrics
    from sklearn.metrics import mean_squared_error, r2_score
    test_mse = mean_squared_error(y_test, best_predictions)
    test_r2 = r2_score(y_test, best_predictions)
    
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    
    # MLflow autologging automatically captures:
    # - All parameters from the grid search
    # - Cross-validation results
    # - Best parameters
    # - Model artifacts
    # - Metrics
    
    print(f"MLflow Run ID: {run.info.run_id}")
    print("Check the MLflow UI to see the GridSearchCV results!")

# Example with different algorithms
with mlflow.start_run(run_name="multi_algorithm_gridsearch") as run:
    print("\nPerforming GridSearchCV with multiple algorithms...")
    
    # Define multiple models and their parameter grids
    models_and_params = [
        {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100],
                'max_depth': [3, 5, None],
                'min_samples_split': [2, 5]
            }
        },
        {
            'model': SVR(),
            'params': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf', 'linear']
            }
        }
    ]
    
    best_overall_score = float('inf')
    best_model_name = ""
    
    for i, model_config in enumerate(models_and_params):
        model = model_config['model']
        params = model_config['params']
        
        # Create and fit GridSearchCV
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=params,
            cv=3,  # Using 3-fold for faster execution
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        # Log which algorithm is being tested
        mlflow.log_param(f"algorithm_{i}", type(model).__name__)
        
        grid_search.fit(X_train, y_train)
        
        # Track the best overall model
        if -grid_search.best_score_ < best_overall_score:
            best_overall_score = -grid_search.best_score_
            best_model_name = type(model).__name__
    
    print(f"Best overall model: {best_model_name}")
    print(f"Best overall CV score: {best_overall_score:.4f}")
    print(f"MLflow Run ID: {run.info.run_id}")
    print("Check the MLflow UI to compare multiple algorithms!")
