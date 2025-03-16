# Prosperity Modelling: A Comprehensive Explanation

This document provides a detailed explanation of the `prosperityModelling.py` script, which implements a complete workflow for analyzing and predicting regional prosperity based on various socioeconomic indicators.

## Table of Contents

1. [Introduction](#introduction)
2. [Data Loading and Structure](#data-loading-and-structure)
3. [Data Preprocessing](#data-preprocessing)
4. [Indicator Labeling](#indicator-labeling)
5. [Exploratory Data Analysis](#exploratory-data-analysis)
6. [Feature Engineering and Dataset Creation](#feature-engineering-and-dataset-creation)
7. [Model Training and Evaluation](#model-training-and-evaluation)
8. [Prediction Generation](#prediction-generation)
9. [Results Visualization](#results-visualization)
10. [Complete Workflow](#complete-workflow)

## Introduction

The `prosperityModelling.py` script implements a comprehensive analysis framework for evaluating regional prosperity across different geographic areas over time. The script processes multiple socioeconomic indicators, categorizes regions based on these indicators, trains machine learning models to predict prosperity, and visualizes the results.

The analysis focuses on four main categories of indicators:

1. Infrastructure indicators (e.g., water access, housing quality)
2. Economic indicators (e.g., minimum wage, GDP)
3. Health indicators (e.g., infant mortality, healthcare facilities)
4. Education indicators (e.g., literacy rate, school participation)

The ultimate goal is to classify regions into prosperity categories (Sejahtera/Prosperous, Menengah/Intermediate, Tidak Sejahtera/Not Prosperous) and identify key factors that contribute to regional prosperity.

## Data Loading and Structure

### Key Function:

- `load_data(file_path, file_type="yearly")`

The script begins by loading data from various sources, handling different file formats and structures. The `load_data` function supports two primary file types:

1. **Yearly data**: Data organized with years as columns
2. **Standard data**: Data with a more traditional structure where each row represents an observation

```python
def load_data(file_path, file_type="yearly"):
    """
    Load data from file path based on file type.

    Parameters:
    file_path (str): Path to the data file
    file_type (str): Type of file - "yearly" or "standard"

    Returns:
    pandas.DataFrame: Loaded data
    """
```

The function handles different file formats (CSV, Excel) and applies initial cleaning, like removing unnecessary rows or columns. Each indicator dataset is loaded individually when needed, using the appropriate file path and structure type.

Why this approach?

- **Flexibility**: Handles different data formats consistently
- **Reusability**: Used across all indicator loading operations
- **Organization**: Clear separation of loading and preprocessing logic

## Data Preprocessing

### Key Functions:

- `preprocess_yearly_data(df, indicator_name, years=None)`
- `preprocess_standard_data(df, indicator_name, years=None)`
- Indicator-specific preprocessing functions (e.g., `preprocess_akses_air_minum`, `preprocess_indeks_pembangunan_manusia`)
- Special case handling functions (e.g., `handle_special_case_amh`, `handle_special_case_apm_apk`)

Each indicator requires specific preprocessing steps to ensure data consistency and quality. The script provides two general preprocessing functions:

1. `preprocess_yearly_data`: Transforms data where years are columns into a long format
2. `preprocess_standard_data`: Processes standard-format data, filtering for relevant years and regions

Additionally, each indicator has a dedicated preprocessing function that handles its unique characteristics. For example:

```python
def preprocess_akses_air_minum(df):
    """Preprocess water access data"""
    return preprocess_yearly_data(df, "akses_air_minum")

def preprocess_kendaraan(df):
    """
    Preprocess vehicle data, splitting into two categories:
    - Two-wheeled vehicles
    - Four-wheeled vehicles
    """
    # Special processing to split vehicle categories
```

Special cases are handled separately:

- `handle_special_case_amh`: Manages literacy rate data with missing years
- `handle_special_case_apm_apk`: Processes school participation rate data with special calculations for missing years

Why this approach?

- **Consistency**: Ensures all indicators are processed in a standardized way
- **Specificity**: Addresses unique characteristics of each indicator
- **Maintainability**: Modular design makes it easy to update or add new indicators

## Indicator Labeling

### Key Functions:

- `label_iqr(df, column, reverse=False)`
- Specialized labeling functions (`label_ipm`, `label_tpt`, `label_stunting`)

After preprocessing, each indicator is labeled to classify regions into prosperity categories. The primary labeling method uses the Interquartile Range (IQR) to create three categories:

```python
def label_iqr(df, column, reverse=False):
    """
    Label data using the IQR method.

    Parameters:
    df (pandas.DataFrame): DataFrame to label
    column (str): Column to use for labeling
    reverse (bool): Whether higher values are worse (True) or better (False)

    Returns:
    pandas.DataFrame: DataFrame with added 'label_sejahtera' column
    """
```

The IQR method:

1. Calculates Q1 (25th percentile) and Q3 (75th percentile)
2. Classifies values > Q3 as "Sejahtera" (or "Tidak Sejahtera" if reverse=True)
3. Classifies values < Q1 as "Tidak Sejahtera" (or "Sejahtera" if reverse=True)
4. Classifies values in between as "Menengah"

Some indicators use specialized labeling functions with domain-specific thresholds:

- `label_ipm`: Human Development Index is labeled based on standard thresholds
- `label_tpt`: Unemployment rate is labeled with specific cutoffs
- `label_stunting`: Child stunting percentage uses WHO-recommended thresholds

Why this approach?

- **Objectivity**: IQR provides a statistical basis for categorization
- **Adaptability**: Special cases can use domain-specific thresholds
- **Interpretability**: Clear definitions for each category

## Exploratory Data Analysis

### Key Functions:

- `plot_indicator_distribution(data, indicator_name, title=None)`
- `plot_indicator_trend(data, indicator_name, title=None)`
- `plot_label_distribution(data, indicator_name, title=None)`
- `plot_regional_comparison(data, indicator_name, year='2023', top_n=10, title=None)`
- `plot_label_trend(data, indicator_name, title=None)`
- `create_correlation_df(all_data, year='2023')`

The script includes several visualization functions to explore the data before modeling:

1. **Distribution Analysis**: Shows how indicator values are distributed
2. **Trend Analysis**: Displays how indicators change over time
3. **Label Distribution**: Visualizes the distribution of prosperity labels
4. **Regional Comparison**: Compares indicator values across regions
5. **Correlation Analysis**: Identifies relationships between indicators

```python
def plot_indicator_distribution(data, indicator_name, title=None):
    """
    Plot the distribution of an indicator.

    Parameters:
    data (dict): Dictionary containing preprocessed data
    indicator_name (str): Name of the indicator to plot
    title (str, optional): Custom title for the plot
    """
```

Why this approach?

- **Comprehensive Understanding**: Multiple perspectives on the data
- **Visual Communication**: Clear visualization of patterns and relationships
- **Data Quality Verification**: Helps identify potential issues before modeling

## Feature Engineering and Dataset Creation

### Key Functions:

- `create_combined_dataset(all_data, year='2023')`
- `prepare_data_for_model(combined_df, target_indicator='indeks_pembangunan_manusia')`
- `prepare_all_years_data(combined_datasets, target_indicator='indeks_pembangunan_manusia')`

Before training models, the script combines individual indicators into a unified dataset:

1. `create_combined_dataset`: Merges all indicators for a specific year
2. `prepare_data_for_model`: Separates features (X) and target variable (y) for modeling
3. `prepare_all_years_data`: Combines data across all years for longitudinal analysis

```python
def prepare_all_years_data(combined_datasets, target_indicator='indeks_pembangunan_manusia'):
    """
    Prepare data for all years for model training.

    Parameters:
    combined_datasets (dict): Dictionary of DataFrames for each year
    target_indicator (str): Indicator to use as target variable

    Returns:
    tuple: X_all (features) and y_all (target) for all years
    """
    X_all = pd.DataFrame()
    y_all = pd.Series(dtype='object')

    for year, df in combined_datasets.items():
        if df is not None:
            X, y = prepare_data_for_model(df, target_indicator)
            if X is not None and y is not None:
                # Add year as a feature - ensure it's numeric
                X['year'] = pd.to_numeric(year, errors='coerce')

                # Append to the combined dataset
                X_all = pd.concat([X_all, X])
                y_all = pd.concat([y_all, y])

    return X_all, y_all
```

This function is critical as it:

1. Combines data from all years into a single dataset
2. Adds the year as a feature (converting from string to numeric)
3. Handles potential errors and missing data

Why this approach?

- **Comprehensive Analysis**: Includes temporal dimension in the analysis
- **Feature Richness**: Incorporates all available indicators
- **Data Integrity**: Ensures consistent handling of missing values

## Model Training and Evaluation

The script trains and evaluates multiple machine learning models to predict regional prosperity:

1. **Data Preparation**:
   - Split data into training and testing sets
   - Standardize features using `StandardScaler`
2. **Model Training**:
   - Train Random Forest Classifier
   - Train Logistic Regression Classifier
3. **Model Evaluation**:
   - Perform k-fold cross-validation
   - Generate classification reports and confusion matrices
   - Compare model performance

Why this approach?

- **Robustness**: Cross-validation ensures reliable performance estimates
- **Comparison**: Multiple models allow for selecting the best approach
- **Comprehensive Metrics**: Classification reports provide detailed performance analysis

## Prediction Generation

### Key Function:

- `generate_predictions_for_all_data(model, all_data, scaler)`

After training, the best model is used to generate predictions for all regions across all years:

```python
def generate_predictions_for_all_data(model, all_data, scaler):
    """
    Generate predictions for all data.

    Parameters:
    model: Trained model
    all_data (dict): Dictionary of preprocessed data
    scaler: Fitted StandardScaler

    Returns:
    pandas.DataFrame: DataFrame with predictions
    """
```

This function:

1. Iterates through each year's combined dataset
2. Creates a feature matrix for each region-year combination
3. Ensures the year column is properly converted to numeric format
4. Standardizes the features using the fitted scaler
5. Generates predictions using the trained model
6. Combines actual and predicted labels into a results DataFrame

Why this approach?

- **Comprehensive Analysis**: Generates predictions for all available data
- **Consistency**: Uses the same preprocessing and scaling approach as in training
- **Rich Output**: Includes both actual and predicted values for comparison

## Results Visualization

### Key Functions:

- `plot_regional_prosperity(predictions, year, based_on='predicted', title=None)`
- `plot_prediction_distribution(predictions)`
- `plot_prediction_accuracy_by_region(predictions)`
- `plot_prosperity_trends(predictions)`
- `plot_feature_importance_prosperity_correlation(model, feature_names, predictions, combined_data)`

The script includes several visualization functions to communicate the results:

1. **Regional Prosperity Map**: Visualizes prosperity levels across regions
2. **Prediction Distribution**: Shows the distribution of predicted prosperity levels
3. **Prediction Accuracy**: Displays model accuracy by region
4. **Prosperity Trends**: Illustrates how prosperity changes over time
5. **Feature Importance**: Identifies the most influential indicators for prosperity

```python
def plot_feature_importance_prosperity_correlation(model, feature_names, predictions, combined_data):
    """
    Plot feature importance and their correlation with prosperity.

    Parameters:
    model: Trained model
    feature_names (list): List of feature names
    predictions (pandas.DataFrame): DataFrame with predictions
    combined_data (dict): Dictionary of combined datasets
    """
```

Why this approach?

- **Actionable Insights**: Identifies key drivers of prosperity
- **Temporal Analysis**: Shows how prosperity evolves over time
- **Spatial Patterns**: Highlights geographic clusters of prosperity

## Complete Workflow

The complete workflow of the script is as follows:

1. **Data Loading**:

   - Load individual indicator datasets directly when needed
   - Handle different file formats and structures using the `load_data` function

2. **Data Preprocessing**:

   - Preprocess each indicator dataset with specific preprocessing functions
   - Handle special cases and missing data
   - Transform data into a consistent format

3. **Indicator Labeling**:

   - Label each indicator using IQR or specialized methods
   - Create prosperity categories for each indicator

4. **Exploratory Data Analysis**:

   - Visualize distributions, trends, and relationships
   - Identify patterns and potential issues

5. **Feature Engineering**:

   - Combine indicators into unified datasets
   - Create features for model training
   - Handle temporal dimension (year as a feature)

6. **Model Training and Evaluation**:

   - Prepare data for modeling
   - Train multiple machine learning models
   - Evaluate and compare model performance

7. **Prediction Generation**:

   - Generate predictions for all regions across all years
   - Combine actual and predicted values

8. **Results Visualization and Interpretation**:
   - Visualize prosperity patterns and trends
   - Identify key drivers of prosperity
   - Generate actionable insights for policy and intervention

## Key Challenges and Solutions

Throughout the script, several challenges were addressed:

1. **Data Inconsistency**:

   - **Challenge**: Different indicators had different formats and structures
   - **Solution**: Specialized preprocessing functions for each indicator

2. **Missing Data**:

   - **Challenge**: Some years or regions had missing data
   - **Solution**: Special case handling and imputation strategies

3. **Year Handling**:

   - **Challenge**: Year was initially represented as a string
   - **Solution**: Explicit conversion to numeric format using `pd.to_numeric`

4. **Feature Compatibility**:
   - **Challenge**: Ensuring all features are present and correctly formatted for prediction
   - **Solution**: Checking for required columns and maintaining consistent order

## Results and Insights

The script produces several key outputs:

1. **Prosperity Classification**:

   - Each region is classified as "Sejahtera" (Prosperous), "Menengah" (Intermediate), or "Tidak Sejahtera" (Not Prosperous)
   - Classifications are based on a comprehensive set of indicators

2. **Key Prosperity Factors**:

   - The most important indicators for predicting prosperity
   - Relationship between indicators and prosperity outcomes

3. **Temporal Patterns**:

   - How prosperity changes over time
   - Regions with improving or declining prosperity

4. **Spatial Patterns**:
   - Geographic clusters of prosperity
   - Regional disparities and similarities

## Conclusion

The `prosperityModelling.py` script implements a comprehensive framework for analyzing regional prosperity using multiple socioeconomic indicators. Through careful data preprocessing, feature engineering, model training, and visualization, the script provides valuable insights into the factors that contribute to regional prosperity.

The modular design of the script allows for easy updates and extensions, making it a valuable tool for ongoing prosperity analysis. By identifying key prosperity factors and regional patterns, the script can inform targeted interventions and policy decisions to improve prosperity across regions.
