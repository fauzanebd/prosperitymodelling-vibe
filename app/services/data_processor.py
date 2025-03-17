import pandas as pd
import numpy as np
from app.models.indicators import INDICATOR_MODELS
from app.models.thresholds import LabelingThreshold
from app.migrations.import_data import manual_labeling
from app import db

def preprocess_indicator_value(indicator_name, value):
    """
    Preprocess an indicator value (e.g., normalize, scale, etc.)
    
    Parameters:
    -----------
    indicator_name : str
        Name of the indicator
    value : float
        Raw value of the indicator
        
    Returns:
    --------
    float
        Preprocessed value of the indicator
    """
    # For simplicity, just return the raw value
    # In practice, you might want to scale/normalize based on the indicator
    return float(value)

def label_indicator_value(indicator_name, value):
    """
    Label a single indicator value as 'Sejahtera', 'Menengah', or 'Tidak Sejahtera'
    
    Parameters:
    -----------
    indicator_name : str
        Name of the indicator
    value : float
        Preprocessed value of the indicator
        
    Returns:
    --------
    str
        Label for the indicator value
    """
    # Check if there's a manual threshold defined
    threshold = LabelingThreshold.query.filter_by(indicator=indicator_name).first()
    
    if threshold and threshold.labeling_method == 'manual':
        # Use manual thresholds by creating a small dataframe with the single value
        df = pd.DataFrame({indicator_name: [value]})
        labels, threshold = manual_labeling(df, indicator_name, indicator_name)
        return labels.iloc[0]
    else:
        # Use IQR method
        # Get all values for this indicator to calculate thresholds
        model_class = INDICATOR_MODELS[indicator_name]
        all_values = [row.value for row in model_class.query.all()]
        
        if not all_values:
            # If no data exists yet, use a default labeling
            return 'Menengah'
        
        # Check if there's a threshold record with IQR method
        is_reverse = False
        if threshold:
            is_reverse = threshold.is_reverse
        else:
            # Fallback to hard-coded list for backward compatibility
            is_reverse = indicator_name in [
                'tingkat_pengangguran_terbuka',
                'penduduk_miskin',
                'kematian_balita',
                'kematian_bayi',
                'kematian_ibu',
                'persentase_balita_stunting'
            ]
        
        # Calculate IQR thresholds
        q1 = np.percentile(all_values, 25)
        q3 = np.percentile(all_values, 75)
        
        if is_reverse:
            if value <= q1:
                return 'Sejahtera'
            elif value >= q3:
                return 'Tidak Sejahtera'
            else:
                return 'Menengah'
        else:
            if value >= q3:
                return 'Sejahtera'
            elif value <= q1:
                return 'Tidak Sejahtera'
            else:
                return 'Menengah'

def get_all_indicator_data(indicator_name):
    """
    Get all data for a specific indicator
    
    Parameters:
    -----------
    indicator_name : str
        Name of the indicator
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing all data for the indicator
    """
    model_class = INDICATOR_MODELS[indicator_name]
    data = model_class.query.all()
    
    if not data:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame([(d.region, d.year, d.value, d.label_sejahtera) for d in data],
                     columns=['wilayah', 'year', indicator_name, 'label_sejahtera'])
    
    return df

def create_combined_dataset(year):
    """
    Create a combined dataset with all indicators for a specific year
    
    Parameters:
    -----------
    year : int
        Year to filter for
        
    Returns:
    --------
    pd.DataFrame
        Combined dataset with all indicators for the specified year
    """
    # Start with a base dataframe containing region names
    base_df = None
    
    # Find a dataframe that has data for the specified year
    for indicator_name in INDICATOR_MODELS:
        model_class = INDICATOR_MODELS[indicator_name]
        if model_class.query.filter_by(year=year).first():
            base_df = pd.DataFrame([(d.region,) for d in model_class.query.filter_by(year=year).all()],
                                  columns=['wilayah'])
            break
    
    if base_df is None:
        print(f"No data found for year {year}")
        return None
    
    # Add each indicator to the base dataframe
    for indicator_name in INDICATOR_MODELS:
        model_class = INDICATOR_MODELS[indicator_name]
        year_data = model_class.query.filter_by(year=year).all()
        
        if year_data:
            # Convert to DataFrame
            indicator_df = pd.DataFrame([(d.region, d.value, d.label_sejahtera) for d in year_data],
                                       columns=['wilayah', indicator_name, f'label_sejahtera_{indicator_name}'])
            
            # Add the indicator value to the base dataframe
            base_df = base_df.merge(
                indicator_df,
                on='wilayah',
                how='left'
            )
    
    return base_df

def prepare_data_for_model(combined_df, target_indicator):
    """
    Prepare data for model training
    
    Parameters:
    -----------
    combined_df : pd.DataFrame
        Combined dataset with all indicators
    target_indicator : str
        Indicator to use as the target variable
        
    Returns:
    --------
    X : pd.DataFrame
        Features for model training
    y : pd.Series
        Target variable for model training
    feature_names : list
        List of feature names
    """
    # Make a copy of the dataframe
    df = combined_df.copy()
    
    # Get the target variable (prosperity label for the specified indicator)
    target_label_col = f'label_sejahtera_{target_indicator}'
    
    # Check if the target column exists
    if target_label_col not in df.columns:
        print(f"Target column {target_label_col} not found in the dataframe")
        return None, None, None
    
    # Create the target variable
    y = df[target_label_col].map({'Sejahtera': 1, 'Menengah': 0, 'Tidak Sejahtera': 0})
    
    # Create the feature matrix
    # Drop non-feature columns (wilayah, label columns, and the target indicator)
    drop_cols = ['wilayah']
    drop_cols.extend([col for col in df.columns if 'label_sejahtera' in col])
    drop_cols.append(target_indicator)
    
    # Keep only numeric columns for features
    X = df.drop(columns=drop_cols, errors='ignore')
    X = X.select_dtypes(include=['number'])
    
    # Handle missing values
    # For simplicity, we'll fill missing values with the mean of each column
    X = X.fillna(X.mean())
    
    # Get feature names
    feature_names = X.columns.tolist()
    
    return X, y, feature_names 