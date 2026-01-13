import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
import mlflow
import random
import os

# This is the new Code!

os.environ['MLFLOW_TRACKING_USERNAME'] = "admin"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "admin0123456789"

mlflow.set_tracking_uri("http://localhost:5000")

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
    n_estimators = random.randint(50, 100)
    max_depth = random.randint(2, 10)
    max_features = random.randint(2, 10)

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
