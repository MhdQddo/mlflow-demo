import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
import mlflow

import argparse

# 1. Initialize the parser
parser = argparse.ArgumentParser(description="A simple greeting script")

# 2. Add the arguments
parser.add_argument("--n_estimators", type=str, help="n_estimators")
parser.add_argument("--max_depth", type=str, help="max_depth")
parser.add_argument("--max_features", type=int, help="max_features")

# 3. Parse the arguments
args = parser.parse_args()

# 4. Access the data
print(f"n_estimators {args.n_estimators}")
print(f"max_depth {args.max_depth}")
print(f"max_features {args.max_features}")

mlflow.set_tracking_uri("http://localhost:5000")
# mlflow.set_experiment('remote-run-exp')

# Enable autologging for all supported libraries 
mlflow.sklearn.autolog()

# Load dataset
db = load_diabetes()
X = db.data
y = db.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Start an MLflow run (optional, autolog will start one if none exists)
with mlflow.start_run() as run:
    # Set the model parameters
    n_estimators = int(args.n_estimators)
    max_depth = int(args.max_depth)
    max_features =int(args.max_features)

    # Create and train the model
    # Autologging captures parameters, metrics (on training set), and the model
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, random_state=42)
    rf.fit(X_train, y_train)

    # Use the model to make predictions (optional, autologging is triggered by .fit())
    predictions = rf.predict(X_test)
    
    # You can still log custom metrics or artifacts manually inside the run
    # mlflow.log_metric("custom_test_score", rf.score(X_test, y_test))
    
    # The run ID can be retrieved for later use
    print(f"MLflow Run ID: {run.info.run_id}")
