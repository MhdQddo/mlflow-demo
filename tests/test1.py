import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
import argparse

# Initialize the parser
parser = argparse.ArgumentParser(description="Test script for MLflow parameterized runs")

# Add the arguments
parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees in the forest (default: 100)")
parser.add_argument("--max_depth", type=int, default=5, help="Maximum depth of the tree (default: 5)")
parser.add_argument("--max_features", type=int, default=5, help="Number of features to consider (default: 5)")

# Parse the arguments
args = parser.parse_args()

# Print the received parameters
print(f"n_estimators: {args.n_estimators}")
print(f"max_depth: {args.max_depth}")
print(f"max_features: {args.max_features}")

# Set the tracking URI
# mlflow.set_tracking_uri("http://localhost:5000")

# Enable autologging
mlflow.sklearn.autolog()

# Load dataset
db = load_diabetes()
X = db.data
y = db.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Start an MLflow run
with mlflow.start_run() as run:
    # Create and train the model with the parsed parameters
    rf = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        max_features=args.max_features,
        random_state=42
    )
    rf.fit(X_train, y_train)

    # Make predictions
    predictions = rf.predict(X_test)

    # Log the parameters explicitly as well
    mlflow.log_params({
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "max_features": args.max_features
    })

    # Log some metrics
    from sklearn.metrics import mean_squared_error, r2_score
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    mlflow.log_metrics({
        "mse": mse,
        "r2_score": r2
    })

    print(f"MLflow Run ID: {run.info.run_id}")
    print(f"MSE: {mse}")
    print(f"R2 Score: {r2}")