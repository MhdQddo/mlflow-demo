# MLflow Demo Project

## Project Overview
This is an MLflow demonstration project that showcases automated logging capabilities for machine learning experiments. The project uses MLflow's autologging feature to automatically capture model parameters, metrics, and artifacts during training of a Random Forest Regressor model on the diabetes dataset.

### Key Components:
- **MLproject**: Defines the project configuration, including conda environment and entry points
- **conda.yaml**: Specifies the conda environment with required dependencies
- **train.py**: Main training script that implements MLflow autologging
- **README.md**: Basic project documentation

### Architecture:
The project implements a simple machine learning pipeline using scikit-learn's Random Forest Regressor trained on the diabetes dataset. It connects to an MLflow tracking server running on localhost:5000 to log experiment data automatically.

## Building and Running

### Prerequisites:
- Conda or Miniconda installed
- MLflow tracking server running on http://localhost:5000

### Setup:
1. Create the conda environment:
   ```bash
   conda env create -f conda.yaml
   ```

2. Activate the environment:
   ```bash
   conda activate mq_env
   ```

### Running the Project:
Execute the training script directly:
```bash
python train.py
```

Or run via MLproject:
```bash
mlflow run .
```

With parameters:
```bash
mlflow run . -P n_estimators=150 -P max_depth=7
```

### Starting MLflow Tracking Server:
If you don't have a tracking server running, start one with:
```bash
mlflow server --host 0.0.0.0 --port 5000
```

## Development Conventions

### MLflow Autologging:
The project demonstrates MLflow's autologging capabilities which automatically capture:
- Model parameters (hyperparameters)
- Training metrics
- Trained model artifacts
- Feature importance (when available)

### Environment Management:
- Dependencies are managed through conda.yaml
- Python 3.10 is specified as the runtime version
- MLflow and scikit-learn are the primary dependencies

### Configuration:
- Tracking URI is hardcoded to "http://localhost:5000"
- Authentication credentials are set via environment variables in the script
- Model hyperparameters can be adjusted via MLproject entry point parameters

## Key Features

1. **Autologging**: Automatically captures training parameters, metrics, and models
2. **Experiment Tracking**: Records each training run with unique run IDs
3. **Model Registry**: Models are automatically registered with MLflow
4. **Parameter Tuning**: Random parameter selection for demonstration purposes

## File Descriptions

- `MLproject`: Project configuration file defining environment and entry points
- `conda.yaml`: Conda environment specification with dependencies
- `train.py`: Main training script implementing MLflow autologging
- `README.md`: Project overview and usage instructions
- `.git/`: Git repository metadata

## Security Notes

The current implementation sets MLflow tracking credentials directly in the source code. For production use, consider using more secure methods for credential management such as:
- Environment variables set outside the application
- MLflow configuration files
- External secret management systems