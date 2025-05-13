# Data Analysis Dashboard

A comprehensive, interactive data analysis and machine learning platform built with Dash and Plotly. This all-in-one dashboard allows users to upload, clean, analyze, visualize, and model data without writing code.

## Features

### Data Import and Management
- Upload and parse CSV and Excel files with or without headers
- Interactive data tables with sorting and filtering
- Export data in various formats (CSV, Excel, JSON)

### Data Cleaning and Preprocessing
- Advanced imputation of missing values (mean, median, mode, KNN)
- Automatic detection and removal of duplicates
- Outlier detection and handling with multiple methods
- Data type conversion and suggestions

### Data Encoding
- Multiple encoding methods (One-Hot, Label, Ordinal)
- Interactive preview of encoded data
- Export of encoded datasets

### Exploratory Data Analysis (EDA)
- Comprehensive summary statistics
- Auto-generated visualizations based on data types
- Interactive plots (histograms, scatter plots, box plots, etc.)
- Correlation analysis and correlation matrices

### Statistical Testing
- Multiple statistical tests (Chi-squared, T-test, ANOVA, etc.)
- Visual representation of test results
- Tabular result display with p-values and statistics

### Regression Analysis
- Linear regression modeling with visualization
- Model equations and performance metrics
- Prediction functionality with confidence intervals

### Machine Learning Prediction
- Random Forest model training and evaluation
- Feature selection and importance
- Make predictions via manual input or file upload
- Visual representation of model performance

### Time Series Analysis
- Time series visualization and decomposition
- Seasonal analysis and trend detection
- Moving averages and forecasting

### Report Generation
- Automated EDA report with insights
- Comprehensive data summary

### Modern UI/UX
- Beautiful dark theme with teal accent colors
- Responsive and intuitive design
- Interactive navigation with sidebar
- Status notifications and error handling

## Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/FRIGUI-ILYES/Dashboard
   cd <Dashboard>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   python app.py
   ```
4. Open your browser and go to [http://127.0.0.1:8050/](http://127.0.0.1:8050/)

## Usage Guide
1. **Import Data**: Start by uploading your dataset (CSV or Excel)
2. **Data Summary**: View basic statistics and information about your data
3. **Data Cleaning**: Handle missing values, duplicates, and outliers
4. **Encoding**: Transform categorical variables into numeric format
5. **Statistics & Visualization**: Explore your data through various plots and visualizations
6. **Statistical Tests**: Perform hypothesis tests to validate assumptions
7. **Regression Analysis**: Create predictive models
8. **ML Prediction**: Train machine learning models and make predictions
9. **Generate Reports**: Create comprehensive EDA reports
10. **Export Results**: Download processed data or analysis results

## Requirements
- Python 3.7+
- See requirements.txt for detailed dependencies

## Contact
For questions or support, contact: ilyes.frigui.ps@gmail.com 
