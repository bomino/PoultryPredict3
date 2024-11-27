# Poultry Weight Predictor Documentation

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Data Requirements](#data-requirements)
4. [Application Structure](#application-structure)
5. [Features and Functionality](#features-and-functionality)
6. [Machine Learning Models](#machine-learning-models)
7. [Model Recommendation System](#model-recommendation-system)
8. [User Guide](#user-guide)
9. [Advanced Features](#advanced-features)
10. [Troubleshooting](#troubleshooting)
11. [API Reference](#api-reference)
12. [Best Practices](#best-practices)
13. [Contributing Guidelines](#contributing-guidelines)

## Overview

The Poultry Weight Predictor is a sophisticated machine learning application that helps poultry farmers and researchers predict poultry weight based on environmental and feeding data. Built with Streamlit and scikit-learn, it offers multiple machine learning models, comprehensive data analysis, and advanced prediction capabilities.

### Key Features
- Multiple machine learning models
- Intelligent model selection
- Interactive data analysis
- Real-time predictions
- Model comparison tools
- Comprehensive documentation
- Export capabilities

## Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Virtual environment (recommended)
- 4GB RAM minimum
- 2GB free disk space

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/bomino/PoultryPredict.git
cd PoultryPredict
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

### System Requirements
- Operating System: Windows 10+, macOS 10.14+, or Linux
- Browser: Chrome, Firefox, or Safari
- Network: Internet connection for initial setup

## Data Requirements

### Input Data Format

The application expects CSV files with the following columns:

| Column Name    | Description                | Unit  | Type    | Valid Range |
|---------------|---------------------------|-------|---------|-------------|
| Int Temp      | Internal Temperature      | °C    | float   | 15-40      |
| Int Humidity  | Internal Humidity         | %     | float   | 30-90      |
| Air Temp      | Air Temperature           | °C    | float   | 10-45      |
| Wind Speed    | Wind Speed               | m/s   | float   | 0-15       |
| Feed Intake   | Feed Intake              | g     | float   | > 0        |
| Weight        | Poultry Weight (target)  | g     | float   | > 0        |

### Data Quality Requirements
1. **Missing Values**
   - Training data: No missing values allowed
   - Prediction data: Features must be complete

2. **Data Types**
   - All values must be numeric
   - No text or categorical data
   - No special characters

3. **Value Ranges**
   - Must be within specified ranges
   - Outliers are detected and flagged
   - Extreme values require validation

4. **Sample Size**
   - Minimum: 2 samples for training
   - Recommended: 50+ samples
   - Optimal: 100+ samples

5. **Data Format**
   - CSV format
   - Comma-separated
   - UTF-8 encoding

## Application Structure

### Directory Layout
```
poultry_weight_predictor/
│
├── app/
│   ├── main.py                  # Application entry point
│   │
│   ├── pages/
│   │   ├── 1_Data_Upload.py       # Data upload and validation
│   │   ├── 2_Data_Analysis.py     # Data analysis and visualization
│   │   ├── 3_Model_Training.py    # Model training and evaluation
│   │   ├── 4_Predictions.py       # Making predictions
│   │   ├── 5_Model_Comparison.py  # Model comparison tools
│   │   └── 6_About.py            # About page
│   │
│   ├── models/
│   │   ├── model_factory.py       # Model creation and management
│   │   ├── polynomial_regression.py# Polynomial regression model
│   │   ├── gradient_boosting.py   # Gradient boosting model
│   │   ├── svr_model.py          # Support vector regression
│   │   └── random_forest.py      # Random forest model
│   │
│   ├── utils/
│   │   ├── data_processor.py      # Data processing utilities
│   │   ├── visualizations.py      # Visualization functions
│   │   ├── model_comparison.py    # Model comparison utilities
│   │   └── validation.py         # Data validation functions
│   │
│   └── config/
│       └── settings.py            # Application settings
│
├── models/                      # Saved models directory
├── data/                       # Sample data directory
└── tests/                      # Unit tests
```

### Component Description
1. **Main Application (main.py)**
   - Application configuration
   - Page routing
   - Session state management
   - Error handling

2. **Pages**
   - Data Upload: File handling and validation
   - Data Analysis: Statistical analysis and visualization
   - Model Training: Model selection and training
   - Predictions: Making and exporting predictions
   - Model Comparison: Performance analysis
   - About: Application information

3. **Models**
   - Model Factory: Central model management
   - Individual model implementations
   - Model persistence
   - Performance metrics

4. **Utilities**
   - Data Processing: Data cleaning and preparation
   - Visualizations: Interactive plots and charts
   - Model Comparison: Comparison tools
   - Validation: Data validation functions

## Features and Functionality

### 1. Data Management
- **Upload System**
  - CSV file upload
  - Data validation
  - Error checking
  - Format verification

- **Data Processing**
  - Automatic cleaning
  - Type conversion
  - Outlier detection
  - Missing value handling

- **Data Analysis**
  - Statistical summaries
  - Distribution analysis
  - Correlation studies
  - Time series visualization

### 2. Model Training
- **Model Selection**
  - Automated recommendations
  - Multiple model support
  - Parameter optimization
  - Cross-validation

- **Training Process**
  - Progress monitoring
  - Performance metrics
  - Error analysis
  - Model persistence

- **Evaluation**
  - Multiple metrics
  - Visual analysis
  - Feature importance
  - Prediction accuracy

### 3. Predictions
- **Input Methods**
  - Manual entry
  - Batch processing
  - Real-time updates

- **Output Options**
  - Point predictions
  - Confidence intervals
  - Uncertainty estimates
  - Export capabilities

### 4. Model Comparison
- **Metrics**
  - Performance comparison
  - Feature importance
  - Error analysis
  - Visual comparison

- **Export Options**
  - Detailed reports
  - Charts and graphs
  - Raw data
  - Analysis results

## Machine Learning Models

### 1. Polynomial Regression
- **Description**: Non-linear regression using polynomial features
- **Parameters**:
  ```python
  {
      'degree': {
          'default': 2,
          'range': (1, 5),
          'type': 'int'
      },
      'fit_intercept': {
          'default': True,
          'type': 'bool'
      },
      'include_bias': {
          'default': True,
          'type': 'bool'
      }
  }
  ```
- **Best for**:
  - Small datasets
  - Simple patterns
  - High interpretability
  - Initial modeling

### 2. Gradient Boosting
- **Description**: Ensemble learning using gradient boosting
- **Parameters**:
  ```python
  {
      'n_estimators': {
          'default': 100,
          'range': (50, 500),
          'type': 'int'
      },
      'learning_rate': {
          'default': 0.1,
          'range': (0.01, 0.3),
          'type': 'float'
      },
      'max_depth': {
          'default': 3,
          'range': (2, 10),
          'type': 'int'
      }
  }
  ```
- **Best for**:
  - Large datasets
  - Complex patterns
  - High accuracy needs
  - Feature importance

### 3. Support Vector Regression
- **Description**: Kernel-based regression for robust predictions
- **Parameters**:
  ```python
  {
      'kernel': {
          'default': 'rbf',
          'options': ['rbf', 'linear', 'poly'],
          'type': 'string'
      },
      'C': {
          'default': 1.0,
          'range': (0.1, 10.0),
          'type': 'float'
      },
      'epsilon': {
          'default': 0.1,
          'range': (0.01, 1.0),
          'type': 'float'
      }
  }
  ```
- **Best for**:
  - Medium datasets
  - Outlier presence
  - Robust predictions
  - Non-linear patterns

### 4. Random Forest
- **Description**: Ensemble learning using multiple decision trees
- **Parameters**:
  ```python
  {
      'n_estimators': {
          'default': 100,
          'range': (50, 500),
          'type': 'int'
      },
      'max_depth': {
          'default': None,
          'range': (3, 20),
          'type': 'int'
      },
      'min_samples_split': {
          'default': 2,
          'range': (2, 10),
          'type': 'int'
      },
      'min_samples_leaf': {
          'default': 1,
          'range': (1, 5),
          'type': 'int'
      },
      'max_features': {
          'default': 'auto',
          'options': ['auto', 'sqrt', 'log2'],
          'type': 'string'
      },
      'bootstrap': {
          'default': True,
          'type': 'bool'
      },
      'oob_score': {
          'default': False,
          'type': 'bool'
      }
  }
  ```
- **Best for**:
  - Medium to large datasets
  - Balanced performance
  - Feature importance
  - Uncertainty estimation

## Model Recommendation System

### Analysis Process
1. **Data Characteristics**
   - Sample size
   - Feature distribution
   - Outlier presence
   - Missing values
   - Data complexity

2. **Problem Requirements**
   - Accuracy needs
   - Interpretability
   - Training time
   - Prediction speed

3. **Resource Constraints**
   - Memory usage
   - Computation power
   - Training time
   - Prediction latency

### Selection Logic
1. **Small Datasets** (< 100 samples)
   - Primary: Polynomial Regression
   - Alternative: SVR
   - Reasoning: Better generalization

2. **Medium Datasets with Outliers**
   - Primary: Random Forest or SVR
   - Alternative: Gradient Boosting
   - Reasoning: Robust predictions

3. **Large Datasets** (> 1000 samples)
   - Primary: Gradient Boosting
   - Alternative: Random Forest
   - Reasoning: High accuracy

4. **Balanced Requirements**
   - Primary: Random Forest
   - Alternative: Gradient Boosting
   - Reasoning: Good all-round performance

### Implementation Example
```python
def suggest_model(data_characteristics: Dict) -> str:
    n_samples = data_characteristics.get('n_samples', 0)
    has_outliers = data_characteristics.get('has_outliers', False)
    complexity = data_characteristics.get('complexity', 'medium')
    
    if n_samples < 100:
        return 'polynomial'
    elif has_outliers and n_samples < 1000:
        return 'random_forest' if complexity == 'high' else 'svr'
    elif complexity == 'high' and n_samples >= 1000:
        return 'gradient_boosting'
    else:
        return 'random_forest'
```

## User Guide

### Getting Started
1. **Data Preparation**
   - Format CSV file
   - Check data quality
   - Validate ranges
   - Handle missing values

2. **Data Upload**
   - Use upload interface
   - Review validation results
   - Check data preview
   - Address any errors

3. **Data Analysis**
   - Review statistics
   - Check distributions
   - Identify patterns
   - Handle outliers

4. **Model Selection**
   - Review recommendations
   - Consider requirements
   - Select model type
   - Configure parameters

5. **Training**
   - Set training parameters
   - Monitor progress
   - Review results
   - Save model

6. **Making Predictions**
   - Choose input method
   - Enter/upload data
   - Get predictions
   - Export results

### Best Practices
1. **Data Quality**
   - Clean thoroughly
   - Handle outliers
   - Use consistent units
   - Validate ranges

2. **Model Selection**
   - Follow recommendations
   - Start simple
   - Compare models
   - Document choices

3. **Training Process**
   - Use cross-validation
   - Monitor metrics
   - Avoid overfitting
   - Save best models

4. **Prediction Usage**
   - Validate inputs
   - Check confidence
   - Monitor accuracy
   - Track performance

## Advanced Features

### 1. Cross-Validation
```python
def perform_cross_validation(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv)
    return {
        'mean_score': scores.mean(),
        'std_score': scores.std(),
        'all_scores': scores
    }
```

### 2. Feature Importance
```python
def get_feature_importance(model, feature_names):
    importances = model.feature_importances_
    return dict(zip(feature_names, importances))
```

### 3. Model Persistence
```python
def save_model(model, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
```

### 4. Uncertainty Estimation
```python
def get_prediction_interval(model, X, percentile=95):
    predictions = []
    for estimator in model.estimators_:
        predictions.append(estimator.predict(X))
    return np.percentile(predictions, [100-percentile, percentile], axis=0)
```

## API Reference

### Core Classes
```python
class ModelFactory:
    def get_model(self, model_type: str, params: dict = None)
    def get_available_models(self) -> dict
    def get_model_params(self, model_type: str) -> dict
    def suggest_model(self, data_characteristics: dict) -> str

class DataProcessor:
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame
    def prepare_features(self, df: pd.DataFrame, test_size: float) -> tuple
    def validate_columns(self, df: pd.DataFrame) -> tuple[bool, list]
    def scale_features(self, X: pd.DataFrame) -> np.ndarray

class BaseModel:
    def train(self, X_train: np.ndarray, y_train: np.ndarray)
    def predict(self, X: np.ndarray) -> np.ndarray
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict
    def get_feature_importance(self) -> dict
```

## Best Practices

### Code Style
- Follow PEP 8 guidelines
- Use meaningful variable names
- Include type hints
- Write comprehensive docstrings
- Add comments for complex logic
- Keep functions focused and small
- Use consistent formatting

### Error Handling
- Use try-except blocks appropriately
- Provide informative error messages
- Log errors for debugging
- Validate inputs
- Handle edge cases
- Implement graceful fallbacks

### Model Development
1. **Data Preparation**
   ```python
   def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
       """
       Prepare data for model training.
       
       Args:
           df (pd.DataFrame): Input dataframe
           
       Returns:
           pd.DataFrame: Processed dataframe
       """
       # Validate input
       if df.empty:
           raise ValueError("Empty dataframe")
           
       # Copy to avoid modifications
       df = df.copy()
       
       # Handle missing values
       df = df.dropna()
       
       # Convert types
       for col in NUMERIC_COLUMNS:
           df[col] = pd.to_numeric(df[col], errors='coerce')
           
       return df
   ```

2. **Model Training**
   ```python
   def train_model(model, X_train, y_train, **kwargs):
       """
       Train model with error handling and logging.
       
       Args:
           model: Model instance
           X_train: Training features
           y_train: Training targets
           **kwargs: Additional parameters
           
       Returns:
           Trained model
       """
       try:
           # Set parameters
           if kwargs:
               model.set_params(**kwargs)
               
           # Train model
           model.train(X_train, y_train)
           
           # Validate training
           if hasattr(model, 'is_trained') and not model.is_trained:
               raise ValueError("Model training failed")
               
           return model
           
       except Exception as e:
           logger.error(f"Training error: {str(e)}")
           raise
   ```

3. **Model Evaluation**
   ```python
   def evaluate_model(model, X_test, y_test):
       """
       Comprehensive model evaluation.
       
       Args:
           model: Trained model
           X_test: Test features
           y_test: Test targets
           
       Returns:
           dict: Evaluation metrics
       """
       metrics = {}
       
       # Basic metrics
       y_pred = model.predict(X_test)
       metrics['mse'] = mean_squared_error(y_test, y_pred)
       metrics['rmse'] = np.sqrt(metrics['mse'])
       metrics['r2'] = r2_score(y_test, y_pred)
       
       # Advanced metrics
       metrics['mae'] = mean_absolute_error(y_test, y_pred)
       metrics['mape'] = mean_absolute_percentage_error(y_test, y_pred)
       
       # Model-specific metrics
       if hasattr(model, 'oob_score_'):
           metrics['oob_score'] = model.oob_score_
           
       return metrics
   ```

### Performance Optimization

1. **Data Processing**
   - Use efficient data structures
   - Optimize memory usage
   - Implement batch processing
   - Cache intermediate results
   - Use appropriate data types

2. **Model Training**
   - Enable early stopping
   - Use appropriate batch sizes
   - Implement parallel processing
   - Monitor resource usage
   - Cache model artifacts

3. **Prediction Pipeline**
   - Optimize feature scaling
   - Implement batch predictions
   - Cache frequent predictions
   - Monitor latency
   - Use efficient data formats

## Troubleshooting

### Common Issues and Solutions

1. **Data Upload Issues**
   - **Problem**: Invalid file format
     ```python
     # Solution: Validate file before processing
     def validate_file(file):
         if not file.name.endswith('.csv'):
             raise ValueError("Invalid file format. Please upload a CSV file.")
     ```

   - **Problem**: Missing columns
     ```python
     # Solution: Check required columns
     def check_columns(df):
         missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
         if missing:
             raise ValueError(f"Missing columns: {missing}")
     ```

2. **Model Training Issues**
   - **Problem**: Poor performance
     ```python
     # Solution: Implement cross-validation
     def validate_performance(model, X, y):
         scores = cross_val_score(model, X, y, cv=5)
         if scores.mean() < PERFORMANCE_THRESHOLD:
             logger.warning("Model performance below threshold")
     ```

   - **Problem**: Overfitting
     ```python
     # Solution: Add regularization
     def add_regularization(model_params):
         if 'max_depth' in model_params:
             model_params['max_depth'] = min(model_params['max_depth'], 10)
         return model_params
     ```

3. **Prediction Issues**
   - **Problem**: Out-of-range predictions
     ```python
     # Solution: Validate predictions
     def validate_predictions(y_pred, valid_range):
         min_val, max_val = valid_range
         invalid = np.logical_or(y_pred < min_val, y_pred > max_val)
         if np.any(invalid):
             logger.warning("Predictions outside valid range detected")
     ```

   - **Problem**: High uncertainty
     ```python
     # Solution: Check prediction confidence
     def check_confidence(model, X):
         if hasattr(model, 'predict_proba'):
             proba = model.predict_proba(X)
             return np.max(proba, axis=1) > CONFIDENCE_THRESHOLD
         return True
     ```

### Error Messages and Solutions

| Error Message | Possible Cause | Solution |
|--------------|---------------|----------|
| "Invalid file format" | Non-CSV file uploaded | Upload .csv file |
| "Missing columns" | Required columns not present | Check file format |
| "Value out of range" | Invalid data values | Validate data ranges |
| "Training failed" | Model training error | Check parameters |
| "High uncertainty" | Low confidence predictions | Validate input data |

## Contributing Guidelines

### Code Contributions

1. **Setting Up Development Environment**
   ```bash
   # Clone repository
   git clone https://github.com/yourusername/poultry-weight-predictor.git
   
   # Create branch
   git checkout -b feature/your-feature-name
   
   # Install development dependencies
   pip install -r requirements-dev.txt
   ```

2. **Code Standards**
   - Follow PEP 8
   - Add type hints
   - Write unit tests
   - Update documentation
   - Include docstrings

3. **Testing**
   ```bash
   # Run tests
   pytest tests/
   
   # Check coverage
   pytest --cov=app tests/
   ```

4. **Documentation**
   - Update relevant docs
   - Add code examples
   - Include docstrings
   - Update README if needed

### Pull Request Process

1. Create feature branch
2. Make changes
3. Add tests
4. Update documentation
5. Submit PR

## License

MIT License - see [LICENSE](LICENSE) for details.

## Support

For support:
- Create GitHub issue
- Email: Bomino@mlawali.com
- Check documentation

---

For more information:
- [README.md](README.md)
- [QUICK_START.md](QUICK_START.md)