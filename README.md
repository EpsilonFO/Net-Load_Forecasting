# Net Energy Demand Forecasting in France

A statistical modeling project for predicting net electrical energy demand (`Net_demand`) in France using time series data and machine learning techniques.

**Authors**: 
- [@LylianChallier](https://github.com/LylianChallier)
- [@EpsilonFO](https://github.com/EpsilonFO)
  
**Date**: March 10, 2025

## Overview

This project focuses on forecasting net electrical energy demand in France using historical data up to the end of 2021. The analysis leverages multiple statistical and machine learning approaches including:

- **GAM (Generalized Additive Models)**: For capturing non-linear and cyclical patterns
- **Random Forest (RF)**: For robust prediction with automated feature importance
- **Quantile Regression Forest (QRF)**: For probabilistic forecasts
- **Linear Regression**: For baseline comparison and variable selection

The primary objective is to accurately predict `Net_demand` (net energy consumption) by identifying and modeling relationships between energy consumption and various temporal, meteorological, and calendar factors.

## Project Structure

```
.
├── Data/
│   ├── train.csv              # Training dataset (historical data)
│   ├── test.csv               # Test dataset (for predictions)
│   └── sample_submission.csv  # Submission template
├── R/
│   └── score.R                # Custom scoring functions (RMSE, MAPE, Pinball Loss)
├── analyse_explo.r            # Exploratory data analysis
├── select_variables.r         # Variable selection procedures
├── modelisation.R             # Final model training and predictions
└── rapport.Rmd                # Full project report (R Markdown)
```

## Dataset Description

### Target Variable
- **Net_demand**: Net electrical energy demand in France (MW)

### Key Features

**Temporal Variables:**
- `Date`, `Time`: Timestamp information
- `Year`, `Month`, `WeekDays`: Calendar components
- `toy` (time of year): Cyclical annual indicator

**Meteorological Variables:**
- `Temp`: Current temperature
- `Temp_s95`, `Temp_s99`: Smoothed temperature (95th and 99th percentiles)
- `Temp_s95_max`, `Temp_s95_min`: Maximum/minimum smoothed temperatures
- `Temp_s99_max`, `Temp_s99_min`: Extreme temperature indicators
- `Wind`, `Wind_weighted`: Wind speed and weighted wind
- `Nebulosity`, `Nebulosity_weighted`: Cloud cover metrics

**Energy Variables:**
- `Load`: Total electrical load
- `Load.1`, `Load.7`: Lagged load (1 day, 7 days)
- `Solar_power`, `Wind_power`: Renewable energy production
- `Net_demand.7`: Lagged net demand (7 days)

**Calendar Variables:**
- `BH` (Bank Holiday): Public holidays indicator
- `BH_before`, `BH_after`: Days before/after holidays
- `Summer_break`, `Christmas_break`: Vacation periods
- `Holiday`, `Holiday_zone_a/b/c`: Regional vacation zones
- `DLS` (Daylight Saving Time): Summer time indicator
- `BH_Holiday`: Combined holiday effect

### Key Relationships

**Load Decomposition:**
```
Load = Net_demand + Solar_power + Wind_power
```

**Temperature Effect:**
- Strong negative correlation between `Net_demand` and `Temp`
- U-shaped relationship: high demand in winter (heating) and summer peaks (cooling)

## Requirements

### R Version
- R >= 4.0.0

### Required Packages

```r
install.packages(c(
  "mgcv",          # GAM models
  "corrplot",      # Correlation visualizations
  "gt",            # Table formatting
  "tidyverse",     # Data manipulation
  "ranger",        # Fast Random Forest implementation
  "randomForest",  # Classic Random Forest
  "xgboost",       # Gradient boosting (optional)
  "yarrr"          # Color palettes
))
```

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/EpsilonFO/Net-Load_Forecasting.git
cd Net-Load_Forecasting
```

### 2. Prepare Data
Ensure the following files are in the `Data/` directory:
- `train.csv`
- `test.csv`
- `sample_submission.csv`

### 3. Install Dependencies
Open R or RStudio and run:
```r
source("install_packages.R")  # Or manually install required packages
```

## Usage

### Quick Start: Complete Pipeline

To run the entire modeling pipeline and generate predictions:

```r
# Load and run the main modeling script
source("modelisation.R")
```

This will:
1. Load and preprocess the data
2. Train GAM and Random Forest models
3. Generate predictions on the test set
4. Save results to `Data/submission_rf.csv`

### Step-by-Step Analysis

#### 1. Exploratory Data Analysis

```r
source("analyse_explo.r")
```

**This script performs:**
- Univariate analysis of time series patterns
- Bivariate correlation analysis
- Visualization of temporal, meteorological, and calendar effects
- Identification of key patterns (seasonality, temperature effects, holiday impacts)

**Key Findings:**
- Clear annual seasonality in net demand
- Strong inverse relationship between demand and temperature
- Significant demand reduction during holidays and summer breaks
- Weekend effect: lower demand on Saturdays and Sundays

#### 2. Variable Selection

```r
source("select_variables.r")
```

**This script implements:**
- **Linear correlation analysis**: Identifies variables with |correlation| > 0.3
- **Backward stepwise selection**: BIC-based variable selection on linear regression
- **Random Forest feature importance**: Permutation-based importance ranking

**Selection Results:**
- **Most important variables**: Load.1, Temp, Temp_s95_max, Temp_s99_max, WeekDays, BH_Holiday
- **Holiday zones** (Holiday_zone_a/b/c) found to be non-significant and excluded

#### 3. Model Training and Evaluation

```r
source("modelisation.R")
```

**Models implemented:**

**GAM Model:**
```r
Net_demand ~ s(Time) + s(toy, bs='cc') + ti(Temp) + ti(Temp_s99) + 
             s(Load.1) + s(Load.7) + ti(Temp_s99, Temp) + 
             WeekDays + BH + te(Temp_s95_max, Temp_s99_max) + 
             Summer_break + Christmas_break + ...
```

**Random Forest Model:**
```r
Net_demand ~ Time + toy + Temp + Temp_s99 + Load.1 + Load.7 + 
             WeekDays + BH + Temp_s95_max * Temp_s99_max + 
             Summer_break + Christmas_break + Wind + ...
```

### Performance Metrics

Models are evaluated using:
- **RMSE** (Root Mean Square Error): Standard error metric
- **MAPE** (Mean Absolute Percentage Error): Percentage-based error
- **Pinball Loss** (quantile=0.8): Asymmetric loss for robust forecasting

Example evaluation results:
```r
# Train/validation split: 2020-2021 for training, 2022+ for evaluation
sel_a <- which(Data0$Year <= 2021)  # Training indices
sel_b <- which(Data0$Year > 2021)   # Evaluation indices
```

### Generating the Report

To compile the full analysis report:

```r
# In RStudio
rmarkdown::render("rapport.Rmd")
```

This generates a comprehensive HTML/PDF report with:
- Complete exploratory analysis
- Variable selection procedures
- Model comparison tables
- Performance visualizations

## Model Details

### GAM (Generalized Additive Model)

**Advantages:**
- Captures non-linear relationships naturally
- Handles cyclical patterns (seasonal effects)
- Interpretable smooth functions
- Built-in regularization with penalization

**Key Components:**
- `s(toy, bs='cc')`: Cyclic cubic spline for annual seasonality
- `ti(Temp)`: Tensor product for temperature effects
- `te(Temp_s95_max, Temp_s99_max)`: Temperature interaction terms

### Random Forest

**Advantages:**
- Robust to outliers and non-linearity
- Automatic feature interaction detection
- No need for explicit feature engineering
- Provides feature importance metrics

**Variants Tested:**
- **RF Complete**: All available variables
- **RF Backward**: Variables selected via linear regression backward selection
- **RF Mixed**: Combines backward selection + RF importance

### Quantile Regression Forest (QRF)

Combines GAM predictions with quantile regression for probabilistic forecasts:
```r
# GAM provides base prediction
# QRF models residuals for uncertainty quantification
QRF_prediction = GAM_prediction + QRF_residual_quantile
```

## Custom Scoring Functions

Located in `R/score.R`:

```r
# Root Mean Square Error
rmse(y, y_hat, digits=0)

# Mean Absolute Percentage Error  
mape(y, y_hat)

# Pinball Loss (for quantile forecasting)
pinball_loss(y, y_hat_quant, quant, output.vect=FALSE)
```

## Results & Submission

### Model Performance

Typical performance on validation set (2022 data):

| Model | RMSE | MAPE | Pinball Loss (0.8) |
|-------|------|------|--------------------|
| GAM   | ~2000 | ~5% | ~650 |
| RF Complete | ~1900 | ~4.8% | ~480 |
| RF Mixed | ~1950 | ~4.9% | ~475 |

### Generating Submissions

Final predictions are saved to:
```
Data/submission_rf.csv
```

Format:
```csv
Date,Net_demand
2022-01-01,45000
2022-01-02,46500
...
```

## Visualization Examples

### Time Series Plot
```r
plot(Data0$Date, Data0$Net_demand, type='l', 
     main="Net Demand Over Time")
```

### Temperature Relationship
```r
plot(Data0$Net_demand ~ Data0$Temp, 
     main="Demand vs Temperature")
```

### Model Predictions vs Actual
```r
plot(Data0$Date[sel_b], predictions, type='l', col='red')
lines(Data0$Date[sel_b], Data0$Net_demand[sel_b], col='blue')
legend("topright", c("Predicted", "Actual"), col=c("red", "blue"), lty=1)
```

## Key Insights

### Exploratory Analysis
1. **Seasonal Pattern**: Clear annual cycle with winter peaks (heating) and summer valleys
2. **Temperature Effect**: Inverse relationship with demand; non-linear with threshold effects
3. **Calendar Effects**: Significant demand reduction during:
   - Weekends (especially Sundays)
   - Public holidays
   - Summer vacation period
4. **Renewable Integration**: Load = Net_demand + Solar + Wind (exact relationship verified)

### Modeling Insights
1. **Most Important Predictors**:
   - Historical load (Load.1, Load.7)
   - Temperature (current and smoothed versions)
   - Calendar variables (WeekDays, holidays)
   
2. **Feature Engineering**:
   - Lagged variables crucial for time series prediction
   - Temperature extremes (Temp_s95, Temp_s99) capture threshold effects
   - Interaction terms improve non-linear modeling

3. **Model Selection**:
   - Random Forest generally outperforms linear and GAM models
   - Variable selection improves interpretability without sacrificing performance
   - Ensemble approaches (GAM + QRF) provide robust uncertainty quantification

## Troubleshooting

### Common Issues

**Missing packages:**
```r
# Check if package is installed
if (!require("mgcv")) install.packages("mgcv")
```

**Memory issues with large datasets:**
```r
# Reduce Random Forest parameters
rf_model <- randomForest(formula, data=train, 
                         ntree=500,        # Reduce trees
                         nodesize=10)      # Increase node size
```

**Date parsing errors:**
```r
# Ensure proper date format
Data0$Date <- as.Date(Data0$Date, format="%Y-%m-%d")
```

### Performance Optimization

For faster computation:
- Use `ranger` instead of `randomForest` (10-100x faster)
- Reduce `ntree` parameter for Random Forest
- Use parallel processing with `doParallel`

## Contributing

This is an academic project. For improvements or suggestions, please contact the authors.

## License

This project is for educational purposes. Please cite appropriately if using any methodology or code.

## References

- **GAM**: Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R*
- **Random Forest**: Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5-32
- **Quantile Regression**: Meinshausen, N. (2006). *Quantile Regression Forests*

---

**Note**: This project analyzes electrical energy demand patterns and builds predictive models for forecasting. The methodology can be adapted to other time series forecasting problems with temporal, meteorological, and calendar features.
