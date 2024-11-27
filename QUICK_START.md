# Quick Start Guide 🚀

Welcome to the Poultry Weight Predictor! This guide will help you get started quickly with the application.

## 1. Installation & Setup ⚙️

### Quick Setup
```bash
# Clone repository
git clone https://github.com/bomino/PoultryPredict3.git
cd PoultryPredict3

# Set up environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Launch application
streamlit run app/main.py
```

## 2. Data Preparation 📊

### Required Format
Your CSV file must include these columns:
```
Int Temp,Int Humidity,Air Temp,Wind Speed,Feed Intake,Weight
29.5,65,28.0,3.2,120,1500
```

### Column Descriptions
| Column       | Unit | Description           | Valid Range |
|-------------|------|----------------------|-------------|
| Int Temp    | °C   | House temperature    | 15-40       |
| Int Humidity| %    | House humidity       | 30-90       |
| Air Temp    | °C   | Outside temperature  | 10-45       |
| Wind Speed  | m/s  | Wind speed          | 0-15        |
| Feed Intake | g    | Daily feed consumed  | > 0         |
| Weight      | g    | Poultry weight      | > 0         |

## 3. Using the Application 🎯

### Step 1: Data Upload
1. Navigate to "📤 Data Upload"
2. Click "Browse files"
3. Select your CSV file
4. Review data preview
5. Check validation results

### Step 2: Data Analysis
1. Go to "📊 Data Analysis"
2. Explore:
   - Time Series Analysis
   - Feature Relationships
   - Outlier Detection
3. Review data quality metrics
4. Identify potential issues

### Step 3: Model Training
1. Visit "🎯 Model Training"
2. Review model recommendation
   - System analyzes your data
   - Suggests best model
   - Provides detailed reasoning
3. Choose model type:
   - Polynomial Regression (simple patterns)
   - Gradient Boosting (complex patterns)
   - SVR (robust predictions)
   - Random Forest (balanced performance)
4. Configure model-specific settings:
   - Basic parameters:
     * Polynomial: degree, fit_intercept
     * Gradient Boosting: n_estimators, learning_rate
     * SVR: kernel, C, epsilon
     * Random Forest: n_estimators, max_depth
   - Advanced options
   - Cross-validation settings
5. Train model
6. Review comprehensive results

### Step 4: Model Selection Guide
Choose based on your needs:

#### Polynomial Regression
- For small datasets (< 100 samples)
- When interpretability is crucial
- For simple pattern detection
- Best with clean, well-structured data

#### Gradient Boosting
- For large datasets (> 1000 samples)
- When maximum accuracy is needed
- For complex pattern recognition
- When computational resources are available

#### Support Vector Regression
- For datasets with outliers
- When robust predictions are needed
- For medium-sized datasets
- When kernel-based learning is beneficial

#### Random Forest
- For balanced performance and interpretability
- When feature importance is crucial
- For medium to large datasets
- When uncertainty estimates are needed
- For robust out-of-box performance
- When ensemble insights are valuable

### Step 5: Make Predictions
1. Go to "🔮 Make Predictions"
2. Choose method:
   - Single Prediction: Manual input
   - Batch Prediction: CSV upload
3. Enter/upload data
4. Get predictions with:
   - Point estimates
   - Confidence intervals (where applicable)
   - Uncertainty bounds (Random Forest)
5. Download results

### Step 6: Compare Models (Optional)
1. Navigate to "📊 Model Comparison"
2. Review metrics:
   - Standard metrics (MSE, RMSE, R²)
   - Model-specific metrics
   - Out-of-bag scores (Random Forest)
3. Compare performances
4. Export reports

## 4. Quick Tips 💡

### Data Quality
✅ Use complete data
✅ Check for outliers
✅ Ensure consistent units
✅ Validate ranges
❌ Avoid missing values
❌ Don't mix units

### Model Selection
✅ Start with recommended model
✅ Consider data characteristics
✅ Review outlier analysis
✅ Check uncertainty needs
❌ Don't ignore recommendations
❌ Don't overcomplicate

### Training Process
✅ Start with default parameters
✅ Enable cross-validation
✅ Monitor performance
✅ Check feature importance
❌ Don't overfit
❌ Don't skip validation

### Making Predictions
✅ Validate inputs
✅ Check confidence metrics
✅ Monitor uncertainty
✅ Document results
❌ Don't extrapolate
❌ Don't ignore warnings

## 5. Common Issues & Solutions 🔧

### Data Upload Issues
- **Issue**: File format error
  - **Fix**: Check CSV format
  - **Fix**: Verify column names

- **Issue**: Validation failures
  - **Fix**: Check data ranges
  - **Fix**: Remove invalid values

### Training Issues
- **Issue**: Poor performance
  - **Fix**: Try recommended model
  - **Fix**: Adjust parameters
  - **Fix**: Enable cross-validation

- **Issue**: High uncertainty
  - **Fix**: Use Random Forest
  - **Fix**: Increase n_estimators
  - **Fix**: Check data quality

### Prediction Issues
- **Issue**: Unreasonable predictions
  - **Fix**: Check input ranges
  - **Fix**: Validate model
  - **Fix**: Review uncertainty

- **Issue**: Low confidence
  - **Fix**: Review data quality
  - **Fix**: Try ensemble methods
  - **Fix**: Increase training data

## 6. Model-Specific Tips 🎯

### Polynomial Regression
- Start with degree=2
- Monitor overfitting
- Scale features properly

### Gradient Boosting
- Tune learning_rate carefully
- Use early_stopping
- Monitor training time

### SVR
- Choose kernel wisely
- Tune C and epsilon
- Scale features

### Random Forest
- Start with 100 estimators
- Consider max_depth
- Enable out-of-bag scoring
- Use feature importance

## 7. Getting Help 🆘

Need assistance?
1. Check error messages
2. Review documentation
3. Check troubleshooting section
4. Open GitHub issue
5. Contact support: Bomino@mlawali.com

## 8. Next Steps 🎓

After getting started:
1. Read full documentation
2. Explore advanced features
3. Optimize your models
4. Automate predictions
5. Export and share results

## 9. Best Practices Summary ✨

1. **Data Management**
   - Keep data organized
   - Document preprocessing
   - Maintain consistent formats
   - Regular backups

2. **Model Development**
   - Follow recommendations
   - Start simple
   - Validate thoroughly
   - Document settings
   - Monitor uncertainty

3. **Production Use**
   - Regular retraining
   - Performance monitoring
   - Result validation
   - Error logging
   - Uncertainty tracking

---

For detailed information, see:
- [Documentation.md](Documentation.md)
- [README.md](README.md)